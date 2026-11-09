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
from routing import (Candidate, CONTEXT_CHARS, CONTEXT_MESSAGES, DECISIONS_URL, RoutingError,
                     candidates_for, decide, decision_input, read_openai_key)
from test_bot import message
from test_weixin import record
from weixin import WeixinBot


CATALOG = [{"slug": "small", "name": "轻量", "efforts": ["low"]},
           {"slug": "strong", "name": "复杂任务", "efforts": ["high"]}]


def response_body(*, reply=None, route="route_1", confidence=0.9):
    answers = []
    if reply is not None:
        answers.append({"name": "should_reply", "type": "predicate", "probability": reply})
    if route is not None:
        answers.append({"name": "route", "type": "choice", "choice": route, "confidence": confidence})
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
    async def test_single_call_combines_reply_and_only_valid_model_effort_pairs(self):
        session = MagicMock()
        session.post.return_value = mock_response(response_body(reply=0.95))
        candidates = [Candidate("small", "low", None, "快速"), Candidate("strong", "high", None, "复杂")]
        result = await decide(session, "sk-test-only", [{"role": "user", "content": "请帮忙排错"}],
                              at=False, candidates=candidates, judge_reply=True)
        self.assertTrue(result.should_reply)
        self.assertEqual(result.candidate, candidates[1])
        session.post.assert_called_once()
        call = session.post.call_args
        self.assertEqual(call.args[0], DECISIONS_URL)
        self.assertFalse(call.kwargs["allow_redirects"])
        self.assertEqual(call.kwargs["timeout"].total, 5)
        self.assertEqual([q["name"] for q in call.kwargs["json"]["questions"]], ["should_reply", "route"])
        self.assertEqual(result.input_tokens, 123)
        self.assertNotIn("sk-test-only", json.dumps(call.kwargs["json"]))

    async def test_gate_rejects_low_scores_before_parsing_unneeded_model_answer(self):
        session = MagicMock()
        session.post.return_value = mock_response({"answers": [
            {"name": "should_reply", "type": "predicate", "probability": 0.79},
            {"name": "route", "type": "refusal"}]})
        result = await decide(session, "key", [], at=False, judge_reply=True,
                              candidates=[Candidate("a", "low", None, ""), Candidate("b", "high", None, "")])
        self.assertFalse(result.should_reply)
        self.assertIsNone(result.candidate)

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

    async def test_unoffered_model_or_effort_is_never_accepted(self):
        candidates = [Candidate("small", "low", None, ""), Candidate("strong", "high", None, "")]
        for value in ("gpt-unapproved", "route_999", True, {}, None):
            session = MagicMock()
            session.post.return_value = mock_response(response_body(route=value))
            with self.assertRaises(RoutingError):
                await decide(session, "key", [], at=True, candidates=candidates)
        session.post.return_value = mock_response(response_body(confidence=0.1))
        result = await decide(session, "key", [], at=True, candidates=candidates)
        self.assertTrue(result.should_reply)
        self.assertIsNone(result.candidate)

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

    async def test_one_valid_candidate_avoids_unnecessary_request(self):
        session = MagicMock()
        candidate = Candidate("only", None, None, "")
        result = await decide(session, "key", [], at=True, candidates=[candidate])
        self.assertEqual(result.candidate, candidate)
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

    def test_candidates_respect_account_capabilities_allowlist_effort_cap_and_attachments(self):
        catalog = [{"slug": "small", "efforts": ["none", "low", "high", "max", "ultra"]},
                   {"slug": "strong", "efforts": ["medium", "xhigh"]}]
        candidates = candidates_for(catalog, "chatgpt", {"auto_model_candidates": {"chatgpt": ["strong"]}})
        self.assertEqual([(c.model, c.effort) for c in candidates], [("strong", "medium")])
        candidates = candidates_for(catalog, "chatgpt", {}, "small")
        self.assertEqual([c.effort for c in candidates], ["none", "low", "high"])
        deep = candidates_for(catalog, "deepseek", {})
        self.assertTrue(any(c.thinking is False and c.effort is None for c in deep))
        self.assertTrue(all(c.effort in (None, "low", "high") for c in deep))
        with self.assertRaises(RoutingError):
            candidates_for(catalog, "chatgpt", {"auto_model_candidates": {"chatgpt": ["missing"]}})
        with self.assertRaises(RoutingError):
            candidates_for([{"slug": str(i), "efforts": []} for i in range(256)], "chatgpt", {})

    def test_config_validation_paths_and_missing_key_do_not_break_manual_startup(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"openai_key_file": "private/openai.key"}), encoding="utf-8")
            config = load_config(path, require_qq=False)
            self.assertEqual(Path(config["openai_key_file"]), Path(folder) / "private/openai.key")
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
        self.config = {"provider": "chatgpt", "chatgpt_model": "small", "data_dir": self.temp.name,
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
        self.assertEqual(call.args[3], "strong")
        self.assertEqual(call.kwargs["reasoning_effort"], "high")
        self.assertEqual(self.bot.model_settings("g")["model"], "small")
        self.assertIsNone(self.bot.chatgpt_effort("g"))
        self.assertIn("strong", self.bot.status("g"))
        self.assertIn("123", self.bot.routing_status("g"))
        self.assertEqual([q["name"] for q in self.bot.http.post.call_args.kwargs["json"]["questions"]], ["route"])

    async def test_manual_addressed_messages_never_call_decisions_or_catalog(self):
        await self.bot.model_command("g", "/模型 自动 关闭", "owner")
        await self.bot.answer(self.history, True, "g")
        self.bot.http.post.assert_not_called()
        self.bot.model_catalog.assert_not_awaited()
        self.assertEqual(self.bot.chatgpt.answer.call_args.args[3], "small")

    async def test_manual_model_can_still_use_reply_gate_and_both_together_use_one_call(self):
        await self.bot.model_command("g", "/模型 手动", "owner")
        await self.bot.answer(self.history, False, "g")
        self.bot.model_catalog.assert_not_awaited()
        questions = self.bot.http.post.call_args.kwargs["json"]["questions"]
        self.assertEqual([q["name"] for q in questions], ["should_reply"])
        self.assertEqual(self.bot.chatgpt.answer.call_args.args[3], "small")
        self.bot.http.post.reset_mock()
        await self.bot.model_command("g", "/模型 自动", "owner")
        await self.bot.answer(self.history, False, "g")
        self.bot.http.post.assert_called_once()
        self.assertEqual(len(self.bot.http.post.call_args.kwargs["json"]["questions"]), 2)

    async def test_skip_and_failed_quick_judgment_never_generate_an_answer(self):
        for response in (mock_response(response_body(reply=0.1)), mock_response({}, 429),
                         mock_response({"answers": [{"name": "should_reply", "type": "refusal"}]})):
            self.bot.http.post.return_value = response
            self.assertEqual(await self.bot.answer(self.history, False, "g"), "[SKIP]")
        self.bot.chatgpt.answer.assert_not_awaited()
        self.assertNotIn("g", self.bot.last_model)

    async def test_failure_or_low_confidence_on_addressed_question_preserves_manual_settings(self):
        self.bot.db.execute("INSERT INTO group_chatgpt_thinking VALUES ('g','low')")
        for response in (mock_response({}, 401), mock_response(response_body(route="not_allowed")),
                         mock_response(response_body(confidence=0.1))):
            self.bot.http.post.return_value = response
            await self.bot.answer(self.history, True, "g")
            call = self.bot.chatgpt.answer.call_args
            self.assertEqual(call.args[3], "small")
            self.assertEqual(call.kwargs["reasoning_effort"], "low")
        self.assertTrue(self.bot.auto_model("g"))

    async def test_gate_can_be_disabled_independently_of_auto_model(self):
        self.bot.frequency_command("g", "/频率 判断 关闭", "owner")
        await self.bot.answer(self.history, False, "g")
        self.assertTrue(self.bot.auto_model("g"))
        self.assertEqual([q["name"] for q in self.bot.http.post.call_args.kwargs["json"]["questions"]], ["route"])

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
        await self.bot.model_command("g", "/模型 strong", "owner")
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
            self.assertEqual(again.model_settings("g")["model"], "strong")
            self.assertEqual(again.chatgpt_effort("g"), "high")
            await again.model_command("g", "/模型 默认", "owner")
            self.assertTrue(again.auto_model("g"))
            self.assertEqual(again.model_settings("g")["model"], "small")
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

    async def test_attachments_keep_existing_model_and_send_only_type_to_decision(self):
        self.bot.model_catalog.return_value = [{"slug": "small", "efforts": ["low", "high"]}, *CATALOG[1:]]
        attachments = [{"kind": "image", "url": "https://private-url", "key": "PRIVATE-KEY"}]
        with patch("bot.input_parts", AsyncMock(return_value=[{"type": "input_image", "image_url": "data:image/png;base64,x"}])):
            await self.bot.answer(self.history, True, "g", attachments=attachments)
        self.assertEqual(self.bot.chatgpt.answer.call_args.args[3], "small")
        self.assertEqual(self.bot.chatgpt.answer.call_args.kwargs["reasoning_effort"], "high")
        sent = json.dumps(self.bot.http.post.call_args.kwargs["json"])
        self.assertNotIn("PRIVATE-KEY", sent)
        self.assertNotIn("private-url", sent)
        self.assertIn("image", sent)
        self.assertEqual(self.history, [{"role": "user", "content": "帮我分析这个报错"}])

    async def test_deepseek_receives_routed_model_and_thinking_without_changing_saved_mode(self):
        self.bot.c.update(provider="deepseek", deepseek_model="deep-fast", deepseek_key="deep-key")
        self.bot.provider = "deepseek"
        self.bot.model_catalog.return_value = [{"slug": "deep-fast", "efforts": ["low", "high"]}]
        self.bot.http.post.side_effect = [mock_response(response_body(route="route_2")), mock_response({
            "choices": [{"message": {"content": "回答", "reasoning_content": "思考"}, "finish_reason": "stop"}]})]
        answer = await self.bot.answer(self.history, True, "g")
        sent = self.bot.http.post.call_args.kwargs["json"]
        self.assertEqual(sent["model"], "deep-fast")
        self.assertEqual(sent["thinking"], {"type": "enabled"})
        self.assertEqual(sent["reasoning_effort"], "high")
        self.assertEqual(answer.reasoning, "思考")
        self.assertEqual(self.bot.thinking("g")["mode"], "off")
        self.bot.chatgpt.answer.assert_not_awaited()

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
            self.assertEqual(wx.chatgpt.answer.call_args.args[3], "strong")
            self.assertEqual([q["name"] for q in wx.http.post.call_args.kwargs["json"]["questions"]], ["route"])
            self.assertIn("strong", await wx.command("/状态"))
        finally:
            wx.db.close()

    async def test_send_diagnostics_use_this_answers_model_and_ignore_stale_calls(self):
        self.bot.qq = AsyncMock(side_effect=aiohttp.ClientConnectionError("failed"))
        self.bot.last_model["g"] = {"provider": "chatgpt", "model": "old-call", "reasoning_effort": "max"}
        actual = {"provider": "chatgpt", "model": "strong", "reasoning_effort": "high"}
        with self.assertRaises(aiohttp.ClientConnectionError):
            await self.bot.send_reply("g", "first", "answer", 1, stage="answer", model_info=actual)
        with self.assertRaises(aiohttp.ClientConnectionError):
            await self.bot.send_reply("g", "second", "model unavailable", 1, stage="answer")
        rows = read_failures(self.folder / "state.db")
        details = {row["message_id"]: row["details"] for row in rows}
        self.assertEqual(details["first"]["model"], "strong")
        self.assertEqual(details["first"]["reasoning_effort"], "high")
        self.assertEqual(details["second"]["model"], "small")


if __name__ == "__main__":
    unittest.main()
