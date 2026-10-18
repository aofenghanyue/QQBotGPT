import asyncio
import contextlib
import datetime as dt
import json
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock

from bot import ApiError, Bot, check_response


def message(mid="m1", group="g1", text="你好", bot=False):
    return {"id": mid, "group_openid": group, "content": text,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "author": {"id": "u1", "username": "测试用户", "bot": bot}}


def group_bot(config, state):
    # Keep the original shared-group behavior under regression coverage.
    return Bot({"owner_only": False, **config}, state)


class BehaviourTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = group_bot({"auto_chat": True, "auto_min_messages": 3, "auto_cooldown_seconds": 600}, ":memory:")

    def tearDown(self):
        self.bot.db.close()

    async def drain(self):
        worker = asyncio.create_task(self.bot.worker())
        await asyncio.wait_for(self.bot.queue.join(), timeout=2)
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker

    async def test_at_has_priority_and_ignores_auto_cooldown(self):
        for i in range(3):
            self.bot.accept("GROUP_MESSAGE_CREATE", message(str(i)))
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message("at"))
        priority, _, job = self.bot.queue.get_nowait()
        self.assertEqual(priority, 0)
        self.assertEqual(job[1], "at")

    async def test_duplicate_at_sends_once(self):
        self.bot.answer = AsyncMock(return_value="你好！")
        self.bot.qq = AsyncMock(return_value={"id": "reply1"})
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message())
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message())
        await self.drain()
        self.bot.qq.assert_awaited_once()

    async def test_full_message_at_detected(self):
        self.bot.bot_id = "bot1"
        data = message()
        data["mentions"] = [{"id": "bot1"}]
        self.bot.accept("GROUP_MESSAGE_CREATE", data)
        self.assertEqual(self.bot.queue.qsize(), 1)
        self.assertEqual(self.bot.queue.get_nowait()[0], 0)

    async def test_full_mode_is_you_with_different_openid_bypasses_auto_limits(self):
        self.bot.bot_id = "ready-id"
        self.bot.c["auto_chat"] = False
        self.bot.answer = AsyncMock(return_value="已收到你的提问")
        self.bot.qq = AsyncMock()
        data = message(text="<@group-scoped-id> 写个新闻稿")
        data["mentions"] = [{"id": "group-scoped-id", "is_you": True}]
        self.bot.accept("GROUP_MESSAGE_CREATE", data)
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", data)
        await self.drain()
        self.bot.answer.assert_awaited_once()
        self.assertTrue(self.bot.answer.call_args.args[1])
        self.bot.qq.assert_awaited_once()

    async def test_other_member_mentions_are_not_treated_as_bot_mentions(self):
        self.bot.bot_id = "ready-id"
        self.bot.c["app_id"] = "123"
        self.bot.c["auto_chat"] = False
        for i, mention in enumerate((
            {"id": "other", "is_you": False}, {"id": "other", "is_you": "false"},
            {"id": "other", "bot": True}, {"id": "other", "bot_appid": "456"})):
            data = message(str(i))
            data["mentions"] = [mention]
            self.bot.accept("GROUP_MESSAGE_CREATE", data)
        self.assertTrue(self.bot.queue.empty())
        self.assertTrue(self.bot.is_at("GROUP_MESSAGE_CREATE", {"mentions": [{"id": "different", "bot_appid": "123"}]}))
        self.assertFalse(self.bot.is_at("GROUP_MESSAGE_CREATE", {"mentions": None}))

    async def test_replayed_old_message_ignored(self):
        data = message()
        data["timestamp"] = "2020-01-01T00:00:00Z"
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", data)
        self.assertTrue(self.bot.queue.empty())

    async def test_bot_messages_and_disallowed_groups_ignored(self):
        self.bot.c["allowed_groups"] = ["g1"]
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message(group="g2"))
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message(bot=True))
        self.assertTrue(self.bot.queue.empty())

    async def test_auto_skip_stays_silent_preserves_quota_and_obeys_cooldown(self):
        self.bot.answer = AsyncMock(return_value="[SKIP]")
        self.bot.qq = AsyncMock()
        for i in range(3):
            self.bot.accept("GROUP_MESSAGE_CREATE", message(str(i)))
        await self.drain()
        self.bot.answer.assert_awaited_once()
        self.bot.qq.assert_not_awaited()
        self.assertEqual(self.bot.db.execute("SELECT COUNT(*) FROM spontaneous").fetchone()[0], 0)
        self.assertTrue(all(m["role"] == "user" for m in self.bot.history["g1"]))
        self.assertEqual(self.bot.last_sent["g1"], 0)
        self.assertIn("保持安静", self.bot.status("g1"))
        for i in range(3, 30):
            self.bot.accept("GROUP_MESSAGE_CREATE", message(str(i)))
        self.assertTrue(self.bot.queue.empty())

    async def test_auto_empty_or_marker_with_explanation_is_not_sent(self):
        self.bot.qq = AsyncMock()
        for i, reply in enumerate(("", "   ", "[skip]", "[SKIP] 不需要接话")):
            self.bot.last_attempt["g1"] = 0
            self.bot.answer = AsyncMock(return_value=reply)
            for j in range(3):
                self.bot.accept("GROUP_MESSAGE_CREATE", message(f"{i}-{j}"))
            await self.drain()
            self.bot.answer.assert_awaited_once()
        self.bot.qq.assert_not_awaited()

    async def test_auto_model_failure_is_silent_without_retry(self):
        self.bot.answer = AsyncMock(side_effect=TimeoutError())
        self.bot.qq = AsyncMock()
        for i in range(3):
            self.bot.accept("GROUP_MESSAGE_CREATE", message(str(i)))
        await self.drain()
        self.bot.answer.assert_awaited_once()
        self.bot.qq.assert_not_awaited()
        self.assertIn("调用失败", self.bot.status("g1"))

    async def test_at_marker_still_replies(self):
        self.bot.answer = AsyncMock(return_value="[SKIP]")
        self.bot.qq = AsyncMock()
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message())
        await self.drain()
        self.bot.qq.assert_awaited_once()
        self.assertTrue(self.bot.qq.call_args.args[2]["content"])
        self.assertNotIn("[SKIP]", self.bot.qq.call_args.args[2]["content"])

    async def test_auto_useful_reply_sends_and_updates_history_quota_status(self):
        self.bot.answer = AsyncMock(return_value="可以使用 dict.fromkeys 去重并保留顺序。")
        self.bot.qq = AsyncMock()
        for i in range(3):
            self.bot.accept("GROUP_MESSAGE_CREATE", message(str(i)))
        await self.drain()
        self.bot.answer.assert_awaited_once()
        self.bot.qq.assert_awaited_once()
        self.assertEqual(self.bot.db.execute("SELECT n FROM spontaneous WHERE grp='g1'").fetchone()[0], 1)
        self.assertEqual(self.bot.history["g1"][-1]["role"], "assistant")
        self.assertIn("已发送回复", self.bot.status("g1"))

    async def test_daily_spontaneous_limit(self):
        self.bot.c["auto_daily_limit"] = 1
        self.bot.answer = AsyncMock(return_value="我补充一句。")
        self.bot.qq = AsyncMock(return_value={"id": "reply1"})
        for i in range(3):
            self.bot.accept("GROUP_MESSAGE_CREATE", message(str(i)))
        await self.drain()
        self.bot.last_attempt["g1"] = 0
        self.bot.last_sent["g1"] = 0
        for i in range(3, 12):
            self.bot.accept("GROUP_MESSAGE_CREATE", message(str(i)))
        self.assertTrue(self.bot.queue.empty())

    async def test_deepseek_error_still_replies_to_at(self):
        self.bot.answer = AsyncMock(side_effect=TimeoutError())
        self.bot.qq = AsyncMock(return_value={"id": "reply1"})
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message())
        await self.drain()
        self.assertIn("稍后", self.bot.qq.call_args.args[2]["content"])

    async def test_group_context_isolation_and_clear(self):
        self.bot.accept("GROUP_MESSAGE_CREATE", message("a", group="g2", text="另一个群的内容"))
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message("b", text="/清空"))
        self.bot.qq = AsyncMock(return_value={"id": "reply1"})
        await self.drain()
        self.assertEqual(len(self.bot.history["g2"]), 1)
        self.assertFalse(any("另一个群" in m["content"] for m in self.bot.history["g1"]))

    async def test_http_200_business_error_rejected(self):
        with self.assertRaises(ApiError):
            check_response(200, {"code": 100016})
        with self.assertRaises(ApiError):
            check_response(200, {"err_code": 40034005})

    async def test_frequency_query_bypasses_cooldown_without_model_or_quota(self):
        self.bot.c["auto_chat"] = False
        self.bot.answer = AsyncMock()
        self.bot.qq = AsyncMock()
        self.bot.accept("GROUP_MESSAGE_CREATE", message(text="/频率"))
        await self.drain()
        self.bot.answer.assert_not_awaited()
        self.assertIn("关闭", self.bot.qq.call_args.args[2]["content"])
        self.assertEqual(self.bot.db.execute("SELECT COUNT(*) FROM spontaneous").fetchone()[0], 0)
        self.assertEqual(len(self.bot.history["g1"]), 0)

    async def test_frequency_requires_verified_role_and_rejects_forgery(self):
        for role in ("", "member", "OWNER"):
            with self.subTest(role=role):
                self.bot.frequency_command("g1", "/频率 1 5 100", role)
                self.assertEqual(self.bot.frequency("g1")["auto_min_messages"], 3)
        self.bot.answer = AsyncMock()
        self.bot.qq = AsyncMock()
        d = message(text="/频率 1 5 100")
        d["author"]["username"] = "群主 owner"
        self.bot.accept("GROUP_MESSAGE_CREATE", d)
        await self.drain()
        self.assertIn("只有群主", self.bot.qq.call_args.args[2]["content"])
        self.bot.answer.assert_not_awaited()

    async def test_frequency_validation_is_atomic(self):
        before = self.bot.frequency("g1")
        for cmd in ("/频率 5 0 100", "/频率 5 60 -1", "/频率 5 60 1001", "/频率 1 abc 100", "/频率 间隔 4"):
            with self.subTest(command=cmd):
                self.assertIn("未更改", self.bot.frequency_command("g1", cmd, "owner"))
                self.assertEqual(before, self.bot.frequency("g1"))

    async def test_frequency_changes_persist_and_are_group_scoped(self):
        with tempfile.TemporaryDirectory() as folder:
            path = folder + "/settings.db"
            first = group_bot({}, path)
            first.frequency_command("g1", "/频率 5 60 100", "owner")
            first.frequency_command("g1", "/频率 间隔 15", "admin")
            first.db.close()
            second = group_bot({}, path)
            try:
                self.assertEqual(second.frequency("g1")["auto_min_messages"], 5)
                self.assertEqual(second.frequency("g1")["auto_cooldown_seconds"], 15)
                self.assertEqual(second.frequency("g1")["auto_daily_limit"], 100)
                self.assertEqual(second.frequency("g2")["auto_min_messages"], 20)
                second.frequency_command("g1", "/频率 默认", "owner")
                self.assertEqual(second.frequency("g1")["auto_cooldown_seconds"], 600)
            finally:
                second.db.close()

    async def test_close_command_cancels_queued_auto_reply_but_keeps_at(self):
        self.bot.answer = AsyncMock(return_value="你好")
        self.bot.qq = AsyncMock()
        for i in range(3):
            self.bot.accept("GROUP_MESSAGE_CREATE", message(str(i)))
        d = message("close", text="/频率 关闭")
        d["author"]["member_role"] = "owner"
        self.bot.accept("GROUP_MESSAGE_CREATE", d)
        await self.drain()
        self.bot.answer.assert_not_awaited()
        self.assertFalse(self.bot.frequency("g1")["auto_chat"])
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message("at"))
        await self.drain()
        self.bot.answer.assert_awaited_once()

    async def test_thinking_permissions_validation_and_persistence(self):
        for role in ("", "member", "OWNER"):
            self.bot.thinking_command("g1", "/思考 开启", role)
            self.assertEqual(self.bot.thinking("g1")["mode"], "off")
        before = self.bot.thinking("g1")
        for command in ("/思考 强度 随便", "/思考 开启 多余", "/思考 乱写"):
            self.assertIn("未更改", self.bot.thinking_command("g1", command, "owner"))
            self.assertEqual(self.bot.thinking("g1"), before)
        with tempfile.TemporaryDirectory() as folder:
            path = folder + "/thinking.db"
            first = group_bot({"thinking_mode": "at"}, path)
            first.thinking_command("g1", "/思考 开启", "admin")
            first.thinking_command("g1", "/思考 强度 最高", "owner")
            first.frequency_command("g1", "/频率 默认", "owner")
            first.db.close()
            second = group_bot({"thinking_mode": "at"}, path)
            try:
                self.assertEqual(second.thinking("g1"), {"mode": "all", "effort": "max"})
                self.assertEqual(second.thinking("g2"), {"mode": "at", "effort": "high"})
                second.frequency_command("g1", "/频率 5 60 100", "owner")
                second.thinking_command("g1", "/思考 默认", "owner")
                self.assertEqual(second.thinking("g1")["mode"], "at")
                self.assertEqual(second.frequency("g1")["auto_min_messages"], 5)
            finally:
                second.db.close()

    async def test_memory_rolls_to_limit_and_restores_after_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = folder + "/memory.db"
            first = group_bot({"auto_chat": False}, path)
            first.memory_command("g1", "/上下文 10", "owner")
            for i in range(12):
                first.accept("GROUP_MESSAGE_CREATE", message(str(i), text=f"内容{i}"))
            first.append_memory("g1", {"role": "assistant", "content": "机器人回答"})
            first.accept("GROUP_MESSAGE_CREATE", message("other", group="g2", text="独立群消息"))
            first.db.close()
            second = group_bot({"auto_chat": False}, path)
            try:
                self.assertEqual(len(second.history["g1"]), 10)
                self.assertTrue(second.history["g1"][0]["content"].endswith("内容3"))
                self.assertEqual(second.history["g1"][-1]["role"], "assistant")
                self.assertIn("独立群消息", second.history["g2"][0]["content"])
                self.assertEqual(second.memory_settings("g1")["limit"], 10)
                self.assertEqual(second.memory_settings("g2")["limit"], 100)
            finally:
                second.db.close()

    async def test_memory_validation_permissions_and_resizing(self):
        for i in range(25):
            self.bot.append_memory("g1", {"role": "user", "content": str(i)})
        before = self.bot.memory_settings("g1")
        for role in ("", "member", "OWNER"):
            self.bot.memory_command("g1", "/记忆 10", role)
            self.assertEqual(self.bot.memory_settings("g1"), before)
        for command in ("/记忆 9", "/记忆 501", "/记忆 abc", "/记忆 过期 -1", "/记忆 过期 10081", "/记忆 长度 10 extra"):
            self.assertIn("未更改", self.bot.memory_command("g1", command, "owner"))
            self.assertEqual(self.bot.memory_settings("g1"), before)
        self.bot.memory_command("g1", "/记忆 长度 10", "admin")
        self.assertEqual(len(self.bot.history["g1"]), 10)
        self.assertEqual(self.bot.history["g1"][0]["content"], "15")
        self.bot.memory_command("g1", "/记忆 200", "owner")
        self.assertEqual(len(self.bot.history["g1"]), 10)
        self.bot.frequency_command("g1", "/频率 5 60 100", "owner")
        self.bot.thinking_command("g1", "/思考 开启", "owner")
        self.bot.memory_command("g1", "/记忆 默认", "owner")
        self.assertEqual(self.bot.memory_settings("g1")["limit"], 100)
        self.assertEqual(self.bot.frequency("g1")["auto_min_messages"], 5)
        self.assertEqual(self.bot.thinking("g1")["mode"], "all")
        stored = json.loads(self.bot.db.execute("SELECT messages FROM group_memory WHERE grp='g1'").fetchone()[0])
        self.assertEqual(len(stored), 10)

    async def test_expiration_removes_persisted_memory_and_status_does_not_refresh_it(self):
        self.bot.memory_command("g1", "/记忆 过期 1", "owner")
        self.bot.append_memory("g1", {"role": "user", "content": "将过期"}, now=time.time()-50)
        before = self.bot.history_time["g1"]
        self.bot.status("g1")
        self.assertEqual(self.bot.history_time["g1"], before)
        self.bot.expire_memory("g1", now=before+61)
        self.assertEqual(len(self.bot.history["g1"]), 0)
        self.assertIsNone(self.bot.db.execute("SELECT messages FROM group_memory WHERE grp='g1'").fetchone())
        self.bot.memory_command("g1", "/记忆 过期 0", "owner")
        self.bot.append_memory("g1", {"role": "user", "content": "不过期"}, now=1)
        self.bot.expire_memory("g1")
        self.assertEqual(len(self.bot.history["g1"]), 1)

    async def test_expired_memory_is_not_restored(self):
        with tempfile.TemporaryDirectory() as folder:
            path = folder + "/expired.db"
            first = group_bot({}, path)
            first.append_memory("g1", {"role": "user", "content": "很早的消息"}, now=time.time()-360*60-1)
            first.append_memory("g2", {"role": "user", "content": "新消息"})
            first.db.close()
            second = group_bot({}, path)
            try:
                self.assertEqual(len(second.history["g1"]), 0)
                self.assertEqual(len(second.history["g2"]), 1)
                self.assertEqual(second.db.execute("SELECT COUNT(*) FROM group_memory").fetchone()[0], 1)
            finally:
                second.db.close()

    async def test_duplicate_normal_then_at_does_not_duplicate_memory(self):
        self.bot.c["auto_chat"] = False
        self.bot.answer = AsyncMock(return_value="答复")
        self.bot.qq = AsyncMock()
        d = message()
        self.bot.accept("GROUP_MESSAGE_CREATE", d)
        self.bot.accept("GROUP_MESSAGE_CREATE", d)
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", d)
        self.assertEqual(self.bot.counts["g1"], 1)
        self.assertEqual(len(self.bot.history["g1"]), 1)
        await self.drain()
        self.bot.answer.assert_awaited_once()
        self.assertEqual(len(self.bot.history["g1"]), 2)

    async def test_clear_requires_admin_and_invalidates_old_queued_context(self):
        self.bot.answer = AsyncMock(return_value="不能重现旧消息")
        self.bot.qq = AsyncMock()
        self.bot.append_memory("g2", {"role": "user", "content": "另一个群"})
        self.bot.append_memory("g1", {"role": "user", "content": "本群旧内容"})
        self.bot.accept("GROUP_MESSAGE_CREATE", message("denied", text="/清空"))
        await self.drain()
        self.assertEqual(len(self.bot.history["g1"]), 1)
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message("pending", text="带着旧内容提问"))
        d = message("clear", text="/清空")
        d["author"]["member_role"] = "admin"
        self.bot.accept("GROUP_MESSAGE_CREATE", d)
        await self.drain()
        self.bot.answer.assert_not_awaited()
        self.assertEqual(len(self.bot.history["g1"]), 0)
        self.assertEqual(len(self.bot.history["g2"]), 1)
        self.assertIsNone(self.bot.db.execute("SELECT messages FROM group_memory WHERE grp='g1'").fetchone())

    async def test_status_and_memory_commands_are_local_and_report_real_counts(self):
        self.bot.answer = AsyncMock()
        self.bot.qq = AsyncMock()
        self.bot.append_memory("g1", {"role": "user", "content": "PRIVATE_CHAT_TEXT"})
        self.bot.append_memory("g1", {"role": "assistant", "content": "PRIVATE_REPLY_TEXT"})
        for i, command in enumerate(("/status", "/状态", "/记忆", "/上下文")):
            self.bot.accept("GROUP_MESSAGE_CREATE", message(str(i), text=command))
        await self.drain()
        self.bot.answer.assert_not_awaited()
        self.assertEqual(self.bot.qq.await_count, 4)
        for call in self.bot.qq.call_args_list:
            text = call.args[2]["content"]
            self.assertIn("2/100条", text)
            self.assertNotIn("PRIVATE_CHAT_TEXT", text)
            self.assertNotIn("PRIVATE_REPLY_TEXT", text)
        status = self.bot.qq.call_args_list[0].args[2]["content"]
        for label in ("模型", "自动接话", "思考模式", "今日", "冷却", "最近模型调用"):
            self.assertIn(label, status)
        self.assertEqual(self.bot.db.execute("SELECT COUNT(*) FROM spontaneous").fetchone()[0], 0)
        self.assertEqual(len(self.bot.history["g1"]), 2)

    async def test_actual_request_respects_current_capacity_and_character_budget(self):
        self.fake_http()
        history = [{"role": "user", "content": str(i)} for i in range(120)]
        await self.bot.answer(history, True, "g1")
        sent = self.bot.http.post.call_args.kwargs["json"]["messages"]
        self.assertEqual(len(sent), 101)  # system + 100 chat messages
        self.assertEqual(sent[1]["content"], "20")
        self.bot.memory_command("g1", "/上下文 10", "owner")
        await self.bot.answer(history, True, "g1")
        self.assertEqual(len(self.bot.http.post.call_args.kwargs["json"]["messages"]), 11)
        self.assertEqual(self.bot.last_model["g1"]["messages"], 10)
        self.bot.memory_command("g1", "/上下文 500", "owner")
        long_history = [{"role": "user", "content": "x"*2000} for _ in range(100)]
        long_history[-1]["content"] = "当前消息"
        await self.bot.answer(long_history, True, "g1")
        sent = self.bot.http.post.call_args.kwargs["json"]["messages"][1:]
        self.assertLessEqual(sum(len(m["content"]) for m in sent), 60000)
        self.assertEqual(sent[-1]["content"], "当前消息")
        self.assertEqual(self.bot.last_model["g1"]["messages"], len(sent))

    async def test_thinking_command_bypasses_model_and_deduplicates(self):
        self.bot.c["auto_chat"] = False
        self.bot.answer = AsyncMock()
        self.bot.qq = AsyncMock()
        d = message(text="/思考 仅@")
        d["author"]["member_role"] = "admin"
        self.bot.accept("GROUP_MESSAGE_CREATE", d)
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", d)
        await self.drain()
        self.bot.answer.assert_not_awaited()
        self.bot.qq.assert_awaited_once()
        self.assertEqual(self.bot.thinking("g1")["mode"], "at")
        self.assertEqual(self.bot.counts["g1"], 0)
        self.assertEqual(len(self.bot.history["g1"]), 0)
        self.assertEqual(self.bot.db.execute("SELECT COUNT(*) FROM spontaneous").fetchone()[0], 0)

    def fake_http(self, content="最终回答", finish="stop"):
        response = MagicMock()
        response.status = 200
        response.json = AsyncMock(return_value={"choices": [{"message": {
            "content": content, "reasoning_content": "PRIVATE_REASONING_MARKER"}, "finish_reason": finish}]})
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=response)
        context.__aexit__ = AsyncMock(return_value=False)
        self.bot.http = MagicMock()
        self.bot.http.post.return_value = context
        self.bot.c["deepseek_key"] = "test-key"

    async def test_thinking_request_modes_and_final_only(self):
        self.fake_http()
        for mode in ("关闭", "仅@", "开启"):
            self.bot.thinking_command("g1", "/思考 " + mode, "admin")
            for at in (True, False):
                with self.subTest(mode=mode, at=at):
                    reply = await self.bot.answer([{"role": "user", "content": "你好"}], at, "g1")
                    self.assertEqual(reply, "最终回答")
                    body = self.bot.http.post.call_args.kwargs["json"]
                    enabled = mode == "开启" or (mode == "仅@" and at)
                    self.assertEqual(body["thinking"]["type"], "enabled" if enabled else "disabled")
                    self.assertEqual("reasoning_effort" in body, enabled)
                    self.assertEqual(body["max_tokens"], 4096 if enabled else 500)
        self.bot.thinking_command("g1", "/思考 强度 max", "admin")
        await self.bot.answer([], True, "g1")
        self.assertEqual(self.bot.http.post.call_args.kwargs["json"]["reasoning_effort"], "max")
        self.assertEqual(self.bot.http.post.call_args.kwargs["json"]["max_tokens"], 8192)
        await self.bot.answer([], True, "g2")
        self.assertEqual(self.bot.http.post.call_args.kwargs["json"]["thinking"]["type"], "disabled")

    async def test_thinking_exhausted_budget_sends_notice_without_retry(self):
        self.fake_http(None, "length")
        self.bot.thinking_command("g1", "/思考 开启", "owner")
        self.bot.qq = AsyncMock()
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message())
        await self.drain()
        self.bot.http.post.assert_called_once()
        sent = self.bot.qq.call_args.args[2]["content"]
        self.assertIn("长度上限", sent)
        self.assertNotIn("PRIVATE_REASONING_MARKER", sent)

    async def test_help_displays_thinking_and_worker_passes_group(self):
        self.bot.answer = AsyncMock(return_value="回答")
        self.bot.qq = AsyncMock()
        self.bot.accept("GROUP_MESSAGE_CREATE", message(text="/help"))
        await self.drain()
        self.assertIn("本群思考模式", self.bot.qq.call_args.args[2]["content"])
        self.bot.answer.assert_not_awaited()
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message("at", group="g2"))
        await self.drain()
        self.assertEqual(self.bot.answer.call_args.args[2], "g2")

    async def test_frequency_command_deduplicates_and_preserves_daily_usage(self):
        day = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date().isoformat()
        self.bot.db.execute("INSERT INTO spontaneous VALUES (?,?,?)", (day, "g1", 2))
        self.bot.db.commit()
        self.bot.qq = AsyncMock()
        self.bot.answer = AsyncMock()
        d = message(text="/频率 1 5 2")
        d["author"]["member_role"] = "admin"
        self.bot.accept("GROUP_MESSAGE_CREATE", d)
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", d)
        await self.drain()
        self.bot.qq.assert_awaited_once()
        self.bot.answer.assert_not_awaited()
        self.assertTrue(self.bot.daily_limit_reached("g1"))
        self.assertIn("1条", self.bot.frequency_status("g1"))


if __name__ == "__main__":
    unittest.main()
