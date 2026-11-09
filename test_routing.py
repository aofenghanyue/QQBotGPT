import asyncio
import contextlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from bot import Bot, load_config
from diagnostics import read_failures
from routing import (Assessment, CONTEXT_CHARS, CONTEXT_MESSAGES, DECISIONS_URL, RoutingError,
                     decide, decision_input, read_openai_key, select_model)
from test_bot import message
from test_weixin import message as weixin_message, record
from weixin import WeixinBot


CATALOG = [{"slug": name, "efforts": ["low", "medium", "high", "xhigh", "max"]}
           for name in ("gpt-6.1-sol", "gpt-6-astra")]
DEEP_CATALOG = [{"slug": name, "efforts": ["low", "high", "max"]}
                for name in ("deepseek-flash", "deepseek-v4-pro")]


def response_body(*, reply=None, domain="stem", difficulty="4", confidence=0.9):
    answers = []
    if reply is not None:
        answers.append({"name": "should_reply", "type": "predicate", "probability": reply})
    answers.extend([{"name": "domain", "type": "choice", "choice": domain, "confidence": confidence},
                    {"name": "difficulty", "type": "choice", "choice": difficulty, "confidence": confidence}])
    return {"answers": answers, "usage": {"input_tokens": 123}}


def mock_response(data, status=200):
    response = MagicMock()
    response.status = status
    response.json = AsyncMock(return_value=data)
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=response)
    manager.__aexit__ = AsyncMock(return_value=False)
    return manager


class DecisionsTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_call_combines_reply_domain_and_six_difficulty_levels(self):
        session = MagicMock()
        session.post.return_value = mock_response(response_body(reply=0.95))
        result = await decide(session, "sk-test-only", [{"role": "user", "content": "请帮忙排错"}],
                              at=False, classify=True, judge_reply=True)
        self.assertTrue(result.should_reply)
        self.assertEqual(result.assessment, Assessment("stem", 4))
        session.post.assert_called_once()
        call = session.post.call_args
        self.assertEqual(call.args[0], DECISIONS_URL)
        self.assertFalse(call.kwargs["allow_redirects"])
        self.assertEqual(call.kwargs["timeout"].total, 5)
        self.assertEqual([q["name"] for q in call.kwargs["json"]["questions"]], ["should_reply", "domain", "difficulty"])
        self.assertEqual([c["value"] for c in call.kwargs["json"]["questions"][2]["choices"]], list("123456"))
        self.assertEqual(result.input_tokens, 123)
        self.assertNotIn("sk-test-only", json.dumps(call.kwargs["json"]))

    async def test_gate_rejects_low_scores_before_parsing_unneeded_model_answer(self):
        session = MagicMock()
        session.post.return_value = mock_response({"answers": [
            {"name": "should_reply", "type": "predicate", "probability": 0.79},
            {"name": "domain", "type": "refusal"}]})
        result = await decide(session, "key", [], at=False, judge_reply=True, classify=True)
        self.assertFalse(result.should_reply)
        self.assertIsNone(result.assessment)

    async def test_refusals_missing_answers_malformed_data_and_invalid_probabilities_fail_closed(self):
        cases = [None, [], {}, {"answers": {}}, {"answers": ["bad"]},
                 {"answers": [{"name": "should_reply", "type": "refusal"}]},
                 response_body(reply=True), response_body(reply=float("nan")),
                 response_body(reply=1.1), response_body(reply=-0.1),
                 {"answers": [{"name": "should_reply", "type": "predicate", "probability": 1}] * 2}]
        for data in cases:
            with self.subTest(data=data):
                session = MagicMock()
                session.post.return_value = mock_response(data)
                with self.assertRaises(RoutingError):
                    await decide(session, "key", [], at=False, judge_reply=True)

    async def test_invalid_classes_levels_and_low_confidence_never_select_a_model(self):
        for field, values in (("domain", ("gpt-unapproved", True, {}, None)),
                              ("difficulty", ("0", "7", "high", 4, True, None))):
            for value in values:
                session = MagicMock()
                session.post.return_value = mock_response(response_body(**{field: value}))
                with self.assertRaises(RoutingError):
                    await decide(session, "key", [], at=True, classify=True)
        session.post.return_value = mock_response(response_body(confidence=0.1))
        result = await decide(session, "key", [], at=True, classify=True)
        self.assertTrue(result.should_reply)
        self.assertIsNone(result.assessment)
        body = response_body()
        body["answers"][1]["confidence"] = 0.1
        session.post.return_value = mock_response(body)
        self.assertIsNone((await decide(session, "key", [], at=True, classify=True)).assessment)

    async def test_http_and_non_json_errors_never_expose_remote_body(self):
        for status in (302, 401, 403, 429, 500):
            session = MagicMock()
            session.post.return_value = mock_response({"error": "SECRET-REMOTE-BODY"}, status)
            with self.assertRaises(RoutingError) as caught:
                await decide(session, "sk-secret", [], at=False, judge_reply=True)
            self.assertEqual(str(caught.exception), f"http_{status}")
        session.post.return_value = mock_response({})
        session.post.return_value.__aenter__.return_value.json.side_effect = ValueError("SECRET")
        with self.assertRaisesRegex(RoutingError, "invalid_json"):
            await decide(session, "key", [], at=False, judge_reply=True)

    async def test_disabled_decisions_avoid_unnecessary_request(self):
        session = MagicMock()
        result = await decide(session, "key", [], at=True)
        self.assertTrue(result.should_reply)
        self.assertIsNone(result.assessment)
        session.post.assert_not_called()


class RoutingConfigTests(unittest.TestCase):
    def test_context_is_bounded_keeps_latest_request_and_does_not_mutate_history(self):
        history = [{"role": "user", "content": "旧资料" * 2000} for _ in range(25)]
        history += [{"role": "assistant", "content": "已经解决"},
                    {"role": "user", "content": "开头" + "中间" * 10000 + "最后的请求"}]
        original = json.dumps(history)
        evidence = json.loads(decision_input(history, at=False, attachment_types=["image"]))
        self.assertLessEqual(len(evidence["messages"]), CONTEXT_MESSAGES)
        self.assertLessEqual(sum(len(m["content"]) for m in evidence["messages"]), CONTEXT_CHARS)
        self.assertTrue(evidence["messages"][-1]["content"].startswith("开头"))
        self.assertTrue(evidence["messages"][-1]["content"].endswith("最后的请求"))
        self.assertIn({"role": "assistant", "content": "已经解决"}, evidence["messages"])
        self.assertEqual(json.dumps(history), original)

    def test_six_math_tiers_use_only_sol_and_astra_with_increasing_effort(self):
        expected = [("gpt-6.1-sol", "low"), ("gpt-6.1-sol", "medium"), ("gpt-6.1-sol", "high"),
                    ("gpt-6-astra", "high"), ("gpt-6-astra", "xhigh"), ("gpt-6-astra", "max")]
        for level, pair in enumerate(expected, 1):
            result = select_model(CATALOG + [{"slug": "gpt-6-luna", "efforts": ["low"]}], Assessment("stem", level), {})
            self.assertEqual((result.model, result.effort), pair)
            self.assertEqual(result.provider, "chatgpt")
            self.assertIsNone(result.thinking)

    def test_daily_tiers_stay_with_deepseek_and_enable_supported_thinking(self):
        expected = [("deepseek-flash", None), ("deepseek-flash", "low"), ("deepseek-flash", "high"),
                    ("deepseek-v4-pro", "high"), ("deepseek-v4-pro", "high"), ("deepseek-v4-pro", "max")]
        for level, pair in enumerate(expected, 1):
            result = select_model(DEEP_CATALOG, Assessment("daily", level), {})
            self.assertEqual((result.model, result.effort), pair)
            self.assertEqual(result.provider, "deepseek")
            self.assertEqual(result.thinking, level > 1)

    def test_live_capabilities_allowlist_and_effort_cap_constrain_policy(self):
        config = {"auto_model_candidates": {"chatgpt": ["gpt-6.1-sol"]}, "auto_model_max_effort": "high"}
        result = select_model(CATALOG, Assessment("stem", 6), config)
        self.assertEqual((result.model, result.effort), ("gpt-6.1-sol", "high"))
        self.assertIn("备选", result.adjustment)
        self.assertIn("上限", result.adjustment)
        result = select_model([{"slug": "gpt-6-astra", "efforts": ["low", "high"]}], Assessment("stem", 6), {})
        self.assertEqual(result.effort, "high")
        for catalog, config in (([{"slug": "gpt-6-luna", "efforts": ["low"]}], {}),
                                (CATALOG, {"auto_model_candidates": {"chatgpt": ["missing"]}}),
                                (CATALOG, {"auto_model_max_effort": "none"}),
                                ([{"slug": "gpt-6-astra", "efforts": []}], {})):
            with self.assertRaises(RoutingError):
                select_model(catalog, Assessment("stem", 6), config)
        result = select_model(DEEP_CATALOG, Assessment("daily", 6), {"auto_model_max_effort": "xhigh"})
        self.assertEqual(result.effort, "high")
        result = select_model(DEEP_CATALOG, Assessment("daily", 6), {"auto_model_max_effort": "none"})
        self.assertFalse(result.thinking)

    def test_attachments_and_live_lookup_use_gpt_even_for_a_daily_task(self):
        result = select_model(CATALOG, Assessment("daily", 1), {}, attachments=True)
        self.assertEqual((result.provider, result.model, result.effort), ("chatgpt", "gpt-6.1-sol", "high"))
        self.assertIn("附件", result.adjustment)
        result = select_model(CATALOG, Assessment("lookup", 2), {})
        self.assertEqual((result.provider, result.model), ("chatgpt", "gpt-6.1-sol"))

    def test_config_validation_paths_and_missing_key_do_not_break_manual_startup(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"openai_key_file": "private/openai.key"}), encoding="utf-8")
            config = load_config(path, require_qq=False)
            self.assertEqual(Path(config["openai_key_file"]).resolve(), (Path(folder) / "private/openai.key").resolve())
            invalid = {"auto_model": ["true", 1, None], "auto_reply_decision": [1, "false"],
                       "decision_timeout_seconds": [0, 16, True, "5", float("nan")],
                       "auto_reply_threshold": [0.49, 1.1, True, float("inf")],
                       "auto_model_max_effort": ["ultra", "unknown"],
                       "auto_model_candidates": [[], {"unknown": ["x"]}, {"chatgpt": []}, {"chatgpt": [None]}],
                       "openai_key_file": [None, "", 3]}
            for key, values in invalid.items():
                for value in values:
                    with self.subTest(key=key, value=value):
                        path.write_text(json.dumps({key: value}), encoding="utf-8")
                        with self.assertRaises(ValueError):
                            load_config(path, require_qq=False)

    def test_key_file_bom_and_environment_priority_without_disclosing_bad_files(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            path = Path(folder) / "openai.key"
            config = {"openai_key_file": str(path)}
            with self.assertRaisesRegex(RoutingError, "key_unavailable"):
                read_openai_key(config)
            path.write_text("\ufeffsk-file\n", encoding="utf-8")
            self.assertEqual(read_openai_key(config), "sk-file")
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-environment"}):
                self.assertEqual(read_openai_key(config), "sk-environment")
            path.write_text("SECRET explanatory text", encoding="utf-8")
            with self.assertRaisesRegex(RoutingError, "key_invalid"):
                read_openai_key(config)


class BotRoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.key_patch = patch.dict(os.environ, {"OPENAI_API_KEY": ""})
        self.key_patch.start()
        (self.folder / "openai.key").write_text("sk-test-only", encoding="utf-8")
        self.config = {"provider": "chatgpt", "chatgpt_model": "gpt-6.1-sol", "data_dir": self.temp.name,
                       "openai_key_file": str(self.folder / "openai.key"),
                       "auto_model": True, "auto_reply_decision": True}
        self.bot = Bot(self.config, str(self.folder / "state.db"))
        self.bot.http = MagicMock()
        self.bot.http.post.return_value = mock_response(response_body(reply=0.95))
        self.bot.model_catalog = AsyncMock(return_value=CATALOG)
        self.bot.chatgpt.models = AsyncMock(return_value=[
            {"slug": m["slug"], "supported_reasoning_levels": m["efforts"]} for m in CATALOG])
        self.bot.chatgpt.answer = AsyncMock(return_value=("生成的回答", {}))
        self.history = [{"role": "user", "content": "帮我分析这个报错"}]

    def tearDown(self):
        self.bot.db.close()
        self.key_patch.stop()
        self.temp.cleanup()

    async def test_auto_uses_selected_pair_without_overwriting_saved_manual_model(self):
        self.assertEqual(await self.bot.answer(self.history, True, "g"), "生成的回答")
        call = self.bot.chatgpt.answer.call_args
        self.assertEqual(call.args[3], "gpt-6-astra")
        self.assertEqual(call.kwargs["reasoning_effort"], "high")
        self.assertEqual(self.bot.model_settings("g")["model"], "gpt-6.1-sol")
        self.assertIsNone(self.bot.chatgpt_effort("g"))
        self.assertIn("gpt-6-astra", self.bot.status("g"))
        self.assertIn("数理技术 L4", self.bot.status("g"))
        self.bot.model_catalog.assert_awaited_once_with("chatgpt")
        self.assertIn("123", self.bot.routing_status("g"))
        self.assertEqual([q["name"] for q in self.bot.http.post.call_args.kwargs["json"]["questions"]], ["domain", "difficulty"])

    async def test_manual_addressed_messages_never_call_decisions_or_catalog(self):
        await self.bot.model_command("g", "/模型 自动 关闭", "owner")
        await self.bot.answer(self.history, True, "g")
        self.bot.http.post.assert_not_called()
        self.bot.model_catalog.assert_not_awaited()
        self.assertEqual(self.bot.chatgpt.answer.call_args.args[3], "gpt-6.1-sol")

    async def test_manual_model_can_still_use_reply_gate_and_both_together_use_one_call(self):
        await self.bot.model_command("g", "/模型 手动", "owner")
        await self.bot.answer(self.history, False, "g")
        self.bot.model_catalog.assert_not_awaited()
        questions = self.bot.http.post.call_args.kwargs["json"]["questions"]
        self.assertEqual([q["name"] for q in questions], ["should_reply"])
        self.assertEqual(self.bot.chatgpt.answer.call_args.args[3], "gpt-6.1-sol")
        self.bot.http.post.reset_mock()
        await self.bot.model_command("g", "/模型 自动", "owner")
        await self.bot.answer(self.history, False, "g")
        self.bot.http.post.assert_called_once()
        self.assertEqual(len(self.bot.http.post.call_args.kwargs["json"]["questions"]), 3)

    async def test_skip_and_failed_quick_judgment_never_generate_an_answer(self):
        for response in (mock_response(response_body(reply=0.1)), mock_response({}, 429),
                         mock_response({"answers": [{"name": "should_reply", "type": "refusal"}]})):
            self.bot.http.post.return_value = response
            self.assertEqual(await self.bot.answer(self.history, False, "g"), "[SKIP]")
        self.bot.chatgpt.answer.assert_not_awaited()
        self.assertNotIn("g", self.bot.last_model)
        self.bot.model_catalog.assert_not_awaited()

    async def test_failure_or_low_confidence_on_addressed_question_preserves_manual_settings(self):
        self.bot.db.execute("INSERT INTO group_chatgpt_thinking VALUES ('g','low')")
        for response in (mock_response({}, 401), mock_response(response_body(domain="not_allowed")),
                         mock_response(response_body(confidence=0.1))):
            self.bot.http.post.return_value = response
            await self.bot.answer(self.history, True, "g")
            call = self.bot.chatgpt.answer.call_args
            self.assertEqual(call.args[3], "gpt-6.1-sol")
            self.assertEqual(call.kwargs["reasoning_effort"], "low")
        self.assertTrue(self.bot.auto_model("g"))

    async def test_gate_can_be_disabled_independently_of_auto_model(self):
        self.bot.frequency_command("g", "/频率 判断 关闭", "owner")
        await self.bot.answer(self.history, False, "g")
        self.assertTrue(self.bot.auto_model("g"))
        self.assertEqual([q["name"] for q in self.bot.http.post.call_args.kwargs["json"]["questions"]], ["domain", "difficulty"])

    async def test_missing_key_is_safe_and_enabling_commands_do_not_change_settings(self):
        (self.folder / "openai.key").unlink()
        await self.bot.model_command("g", "/模型 手动", "owner")
        self.bot.frequency_command("g", "/频率 判断 关闭", "owner")
        self.assertIn("未更改", await self.bot.model_command("g", "/模型 自动", "owner"))
        self.assertIn("未更改", self.bot.frequency_command("g", "/频率 判断 开启", "owner"))
        self.assertFalse(self.bot.auto_model("g"))
        self.assertFalse(self.bot.frequency("g")["auto_reply_decision"])
        await self.bot.answer(self.history, True, "g")
        await self.bot.answer(self.history, True, "other")
        self.assertEqual(await self.bot.answer(self.history, False, "other"), "[SKIP]")
        self.bot.http.post.assert_not_called()

    async def test_settings_are_owner_controlled_isolated_persistent_and_explicit_changes_exit_auto(self):
        self.assertIn("绑定账号", await self.bot.model_command("g", "/模型 手动", ""))
        self.assertIn("绑定账号", self.bot.frequency_command("g", "/频率 判断 关闭", ""))
        self.assertTrue(self.bot.auto_model("g"))
        await self.bot.model_command("g", "/模型 gpt-6-astra", "owner")
        self.assertFalse(self.bot.auto_model("g"))
        await self.bot.model_command("g", "/模型 自动", "owner")
        self.bot.reasoning_display_command("g", "/思考 显示", "owner")
        self.assertTrue(self.bot.auto_model("g"))
        await self.bot.chatgpt_thinking_command("g", "/思考 高", "owner")
        self.assertFalse(self.bot.auto_model("g"))
        self.bot.frequency_command("g", "/频率 判断 关闭", "owner")
        again = Bot(self.config, str(self.folder / "state.db"))
        try:
            self.assertFalse(again.auto_model("g"))
            self.assertTrue(again.auto_model("private:someone"))
            self.assertFalse(again.frequency("g")["auto_reply_decision"])
            self.assertEqual(again.model_settings("g")["model"], "gpt-6-astra")
            self.assertEqual(again.chatgpt_effort("g"), "high")
            await again.model_command("g", "/模型 默认", "owner")
            self.assertTrue(again.auto_model("g"))
            self.assertEqual(again.model_settings("g")["model"], "gpt-6.1-sol")
        finally:
            again.db.close()

    async def test_catalogue_timeout_is_bounded_and_outer_cancellation_never_falls_back(self):
        cancelled = asyncio.Event()

        async def blocked(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        self.bot.c["decision_timeout_seconds"] = 0.02
        self.bot.model_catalog.side_effect = blocked
        await self.bot.answer(self.history, True, "g")
        self.assertTrue(cancelled.is_set())
        self.assertIn("沿用手动", self.bot.routing_status("g"))
        self.bot.chatgpt.answer.reset_mock()
        self.bot.c["decision_timeout_seconds"] = 5
        task = asyncio.create_task(self.bot.answer(self.history, True, "g"))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.bot.chatgpt.answer.assert_not_awaited()

    async def test_attachments_route_to_gpt_from_deepseek_and_send_only_type_to_decision(self):
        self.bot.c.update(provider="deepseek", deepseek_model="deepseek-flash")
        self.bot.provider = "deepseek"
        self.bot.http.post.return_value = mock_response(response_body(domain="daily", difficulty="1"))
        attachments = [{"kind": "image", "url": "https://private-url", "key": "PRIVATE-KEY"}]
        with patch("bot.input_parts", AsyncMock(return_value=[{"type": "input_image", "image_url": "data:image/png;base64,x"}])):
            await self.bot.answer(self.history, True, "g", attachments=attachments)
        self.assertEqual(self.bot.chatgpt.answer.call_args.args[3], "gpt-6.1-sol")
        self.assertEqual(self.bot.chatgpt.answer.call_args.kwargs["reasoning_effort"], "high")
        sent = json.dumps(self.bot.http.post.call_args.kwargs["json"])
        self.assertNotIn("PRIVATE-KEY", sent)
        self.assertNotIn("private-url", sent)
        self.assertIn("image", sent)
        self.assertEqual(self.history, [{"role": "user", "content": "帮我分析这个报错"}])

    async def test_qq_auto_mode_accepts_attachments_with_saved_deepseek_provider(self):
        self.bot.c.update(provider="deepseek", deepseek_model="deepseek-flash", owner_only=False)
        self.bot.provider = "deepseek"
        self.bot.qq = AsyncMock()
        media = [{"kind": "image"}]
        with patch("bot.qq_media", return_value=(media, [], [])), \
                patch.object(self.bot.attachments, "prepare", return_value=(media, None)), \
                patch("bot.input_parts", AsyncMock(return_value=[{"type": "input_image", "image_url": "data:image/png;base64,x"}])):
            self.bot.accept("GROUP_AT_MESSAGE_CREATE", message("auto-image", text="看图解释"))
            worker = asyncio.create_task(self.bot.worker())
            try:
                await asyncio.wait_for(self.bot.queue.join(), 2)
            finally:
                worker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker
        self.bot.chatgpt.answer.assert_awaited_once()
        self.assertEqual(self.bot.last_model["g1"]["provider"], "chatgpt")

    async def test_daily_question_switches_from_gpt_to_deepseek_without_changing_saved_mode(self):
        self.bot.c.update(deepseek_model="deepseek-flash", deepseek_key="deep-key")
        self.bot.model_catalog.return_value = DEEP_CATALOG
        self.bot.http.post.side_effect = [mock_response(response_body(domain="daily", difficulty="3")), mock_response({
            "choices": [{"message": {"content": "回答", "reasoning_content": "思考"}, "finish_reason": "stop"}]})]
        answer = await self.bot.answer(self.history, True, "g")
        sent = self.bot.http.post.call_args.kwargs["json"]
        self.assertEqual(sent["model"], "deepseek-flash")
        self.assertEqual(sent["thinking"], {"type": "enabled"})
        self.assertEqual(sent["reasoning_effort"], "high")
        self.assertEqual(answer.reasoning, "思考")
        self.assertEqual(self.bot.thinking("g")["mode"], "off")
        self.bot.chatgpt.answer.assert_not_awaited()
        self.bot.model_catalog.assert_awaited_once_with("deepseek")
        self.assertEqual(self.bot.last_model["g"]["provider"], "deepseek")
        self.assertEqual(self.bot.model_settings("g")["provider"], "chatgpt")
        self.assertIn("闲聊生活 L3", self.bot.routing_status("g"))

    async def test_math_switches_from_saved_deepseek_to_astra_and_manual_mode_restores_deepseek(self):
        self.bot.c.update(provider="deepseek", deepseek_model="deepseek-flash")
        self.bot.provider = "deepseek"
        self.bot.http.post.return_value = mock_response(response_body(difficulty="6"))
        await self.bot.answer(self.history, True, "g")
        self.assertEqual(self.bot.chatgpt.answer.call_args.args[3], "gpt-6-astra")
        self.assertEqual(self.bot.chatgpt.answer.call_args.kwargs["reasoning_effort"], "max")
        self.assertEqual(self.bot.model_timeout("g"), 150)
        self.assertEqual(self.bot.model_settings("g")["provider"], "deepseek")
        await self.bot.model_command("g", "/模型 手动", "owner")
        self.bot.http.post.reset_mock()
        selected = await self.bot.route_answer(self.history, True, "g")
        self.assertEqual((selected["provider"], selected["model"]), ("deepseek", "deepseek-flash"))
        self.bot.http.post.assert_not_called()

    async def test_unavailable_target_provider_falls_back_without_overwriting_manual_selection(self):
        self.bot.http.post.return_value = mock_response(response_body(domain="daily", difficulty="2", reply=0.95))
        self.bot.model_catalog.side_effect = RoutingError("no_candidates")
        await self.bot.answer(self.history, True, "g")
        self.assertEqual(self.bot.chatgpt.answer.call_args.args[3], "gpt-6.1-sol")
        self.assertEqual(self.bot.last_routing["g"]["domain"], "daily")
        self.assertIn("沿用手动", self.bot.routing_status("g"))
        self.assertEqual(await self.bot.answer(self.history, False, "g"), "[SKIP]")

    async def test_live_lookup_keeps_search_tools_on_gpt(self):
        self.bot.http.post.return_value = mock_response(response_body(domain="lookup", difficulty="2"))
        await self.bot.answer(self.history, True, "g")
        self.assertEqual(self.bot.chatgpt.answer.call_args.args[3], "gpt-6.1-sol")
        self.assertTrue(self.bot.chatgpt.answer.call_args.kwargs["web_search"])
        self.assertIn("联网查询 L2", self.bot.routing_status("g"))

    async def test_unbound_and_cooldown_block_decisions_and_skips_do_not_write_assistant_memory(self):
        self.bot.accept("GROUP_MESSAGE_CREATE", message("unbound"))
        self.assertTrue(self.bot.queue.empty())
        self.bot.c.update(owner_only=False, auto_chat=True, auto_min_messages=1, auto_cooldown_seconds=5)
        self.bot.qq = AsyncMock()
        self.bot.http.post.return_value = mock_response(response_body(reply=0.1))
        self.bot.accept("GROUP_MESSAGE_CREATE", message("first"))
        worker = asyncio.create_task(self.bot.worker())
        try:
            await asyncio.wait_for(self.bot.queue.join(), 2)
            self.bot.accept("GROUP_MESSAGE_CREATE", message("cooldown"))
            await asyncio.wait_for(self.bot.queue.join(), 2)
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
        self.bot.http.post.assert_called_once()
        self.bot.chatgpt.answer.assert_not_awaited()
        self.bot.qq.assert_not_awaited()
        self.assertFalse(any(m["role"] == "assistant" for m in self.bot.history["g1"]))

    async def test_weixin_uses_auto_routing_without_group_reply_gate(self):
        (self.folder / "wx").mkdir()
        wx = WeixinBot(self.config, self.folder / "wx", record())
        try:
            wx.http = self.bot.http
            wx.model_catalog = self.bot.model_catalog
            wx.chatgpt.answer = AsyncMock(return_value=("微信回答", {}))
            await wx.command("/模型 自动")
            self.assertEqual(await wx.answer(self.history, True, wx.scope), "微信回答")
            self.assertEqual(wx.chatgpt.answer.call_args.args[3], "gpt-6-astra")
            self.assertEqual([q["name"] for q in wx.http.post.call_args.kwargs["json"]["questions"]], ["domain", "difficulty"])
            self.assertIn("gpt-6-astra", await wx.command("/状态"))
        finally:
            wx.db.close()

    async def test_weixin_auto_mode_crosses_providers_and_preserves_manual_settings(self):
        folder = self.folder / "wx"
        folder.mkdir()
        wx = WeixinBot({**self.config, "deepseek_key": "deep-key"}, folder, record())
        try:
            wx.http = self.bot.http
            wx.model_catalog = AsyncMock(side_effect=lambda provider: DEEP_CATALOG if provider == "deepseek" else CATALOG)
            wx.chatgpt.answer = AsyncMock(return_value=("数学回答", {}))
            wx.http.post.side_effect = [
                mock_response(response_body(domain="daily", difficulty="1")),
                mock_response({"choices": [{"message": {"content": "你好"}, "finish_reason": "stop"}]}),
                mock_response(response_body(domain="stem", difficulty="2")),
            ]
            await wx.answer([{"role": "user", "content": "你好"}], True, wx.scope)
            self.assertEqual(wx.last_model[wx.scope]["provider"], "deepseek")
            await wx.answer([{"role": "user", "content": "解释导数"}], True, wx.scope)
            self.assertEqual(wx.last_model[wx.scope]["provider"], "chatgpt")
            self.assertEqual(wx.chatgpt.answer.call_args.kwargs["reasoning_effort"], "medium")
            self.assertEqual(wx.model_settings(wx.scope)["model"], "gpt-6.1-sol")
        finally:
            wx.db.close()

    async def test_weixin_auto_mode_accepts_attachments_with_saved_deepseek_provider(self):
        folder = self.folder / "wx"
        folder.mkdir()
        wx = WeixinBot({**self.config, "provider": "deepseek", "deepseek_model": "deepseek-flash"}, folder, record())
        try:
            wx.http = self.bot.http
            wx.model_catalog = AsyncMock(return_value=CATALOG)
            wx.chatgpt_client().answer = AsyncMock(return_value=("看图回答", {}))
            api = AsyncMock()
            media = [{"kind": "image"}]
            with patch("weixin.weixin_media", return_value=("看图解释", media, [], [])), \
                    patch.object(wx.attachments, "prepare", return_value=(media, None)), \
                    patch("bot.input_parts", AsyncMock(return_value=[{"type": "input_image", "image_url": "data:image/png;base64,x"}])):
                wx.ingest({"msgs": [weixin_message("auto-image")]})
                await wx.process_pending(api)
            wx.chatgpt.answer.assert_awaited_once()
            self.assertEqual(wx.last_model[wx.scope]["provider"], "chatgpt")
            self.assertEqual(wx.model_settings(wx.scope)["provider"], "deepseek")
        finally:
            wx.db.close()

    async def test_send_diagnostics_use_this_answers_model_and_ignore_stale_calls(self):
        self.bot.qq = AsyncMock(side_effect=aiohttp.ClientConnectionError("failed"))
        self.bot.last_model["g"] = {"provider": "chatgpt", "model": "old-call", "reasoning_effort": "max"}
        actual = {"provider": "chatgpt", "model": "gpt-6-astra", "reasoning_effort": "high"}
        with self.assertRaises(aiohttp.ClientConnectionError):
            await self.bot.send_reply("g", "first", "answer", 1, stage="answer", model_info=actual)
        with self.assertRaises(aiohttp.ClientConnectionError):
            await self.bot.send_reply("g", "second", "model unavailable", 1, stage="answer")
        rows = read_failures(self.folder / "state.db")
        details = {row["message_id"]: row["details"] for row in rows}
        self.assertEqual(details["first"]["model"], "gpt-6-astra")
        self.assertEqual(details["first"]["reasoning_effort"], "high")
        self.assertEqual(details["second"]["model"], "gpt-6.1-sol")


if __name__ == "__main__":
    unittest.main()
