import asyncio
import contextlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock

from bot import Bot
from chatgpt import ChatGPTError
from test_personal import event


CATALOG = [
    {"slug": "model-a", "display_name": "A", "supported_reasoning_levels": [{"effort": x} for x in ("low", "high", "max", "ultra")]},
    {"slug": "model-b", "display_name": "B", "supported_reasoning_levels": [{"effort": "low"}]},
]


class GroupAccessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = Bot({"group_members_can_chat": True, "binding_code": "code", "allowed_groups": ["old-group"]}, ":memory:")
        self.bot.answer = AsyncMock(return_value="回答")
        self.bot.qq = AsyncMock()

    def tearDown(self):
        self.bot.db.close()

    async def drain(self):
        task = asyncio.create_task(self.bot.worker())
        try:
            await asyncio.wait_for(self.bot.queue.join(), 2)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def test_shared_group_allows_member_mentions_but_not_settings_or_private_access(self):
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", event("unbound", "hello", user="other"))
        self.assertTrue(self.bot.queue.empty())
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", event("bind", "/绑定 code"))
        await self.drain()
        self.bot.accept("GROUP_MESSAGE_CREATE", event("ordinary", "unaddressed text", user="other"))
        self.bot.accept("C2C_MESSAGE_CREATE", event("private", "private text", user="other"))
        self.assertTrue(self.bot.queue.empty())
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", event("member", "member question", user="other"))
        await self.drain()
        self.bot.answer.assert_awaited_once()
        before = list(self.bot.history["group"])
        self.bot.model_catalog = AsyncMock()
        for i, command in enumerate(("/模型 deepseek another-model", "/思考 高", "/清空", "/记忆 10", "/频率 开启")):
            incoming = event(str(i), command, user="other")
            incoming["author"]["member_role"] = "admin"
            self.bot.accept("GROUP_AT_MESSAGE_CREATE", incoming)
            await self.drain()
            self.assertIn("绑定账号", self.bot.qq.call_args.args[2]["content"])
        self.bot.answer.assert_awaited_once()
        self.bot.model_catalog.assert_not_awaited()
        self.assertEqual(list(self.bot.history["group"]), before)
        self.assertFalse(self.bot.frequency("group")["auto_chat"])
        self.assertEqual(self.bot.memory_settings("group")["limit"], 100)

    async def test_new_group_code_is_shared_across_processes_single_use_and_group_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "state.db")
            running = Bot({"group_members_can_chat": True, "allowed_groups": ["old"]}, path)
            issuer = Bot({}, path)
            code = issuer.issue_group_binding()
            issuer.db.close()
            try:
                stored = running.db.execute("SELECT digest FROM binding_token").fetchone()[0]
                self.assertNotIn(code, stored)
                running.accept("C2C_MESSAGE_CREATE", event("private-bind", "/绑定 " + code))
                self.assertIsNone(running.db.execute("SELECT 1 FROM owners").fetchone())
                running.accept("GROUP_AT_MESSAGE_CREATE", event("bind", "/绑定 " + code, group="new-group"))
                self.assertEqual(running.db.execute("SELECT scope,user_id FROM owners").fetchone(), ("group:new-group", "self"))
                self.assertIsNone(running.db.execute("SELECT 1 FROM binding_token").fetchone())
                running.accept("GROUP_AT_MESSAGE_CREATE", event("replay", "/绑定 " + code, group="second-group", user="other"))
                self.assertEqual(running.db.execute("SELECT COUNT(*) FROM owners").fetchone()[0], 1)
                self.assertFalse(running.history["new-group"])
            finally:
                running.db.close()

    async def test_new_group_code_cannot_overwrite_binding_and_expires(self):
        self.bot.db.execute("INSERT INTO owners VALUES ('group:group', 'self')")
        self.bot.db.commit()
        code = self.bot.issue_group_binding()
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", event("overwrite", "/绑定 " + code, user="other"))
        self.assertEqual(self.bot.db.execute("SELECT user_id FROM owners").fetchone()[0], "self")
        self.assertIsNotNone(self.bot.db.execute("SELECT 1 FROM binding_token").fetchone())
        self.bot.db.execute("UPDATE binding_token SET expires=?", (time.time() - 1,))
        self.bot.db.commit()
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", event("expired", "/绑定 " + code, group="new"))
        self.assertEqual(self.bot.db.execute("SELECT COUNT(*) FROM owners").fetchone()[0], 1)


class ModelSettingsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = {"provider": "chatgpt", "chatgpt_model": "model-a", "data_dir": self.temp.name}
        self.path = str(Path(self.temp.name) / "state.db")
        self.bot = Bot(self.config, self.path)
        self.bot.chatgpt.models = AsyncMock(return_value=CATALOG)
        self.bot.http = MagicMock()

    def tearDown(self):
        self.bot.db.close()
        self.temp.cleanup()

    async def test_model_and_effort_are_per_conversation_persistent_and_owner_controlled(self):
        self.assertIn("绑定账号", await self.bot.model_command("one", "/模型 model-b", ""))
        self.assertIn("绑定账号", await self.bot.chatgpt_thinking_command("one", "/思考 高", ""))
        self.bot.chatgpt.models.assert_not_awaited()
        self.assertIn("已保存", await self.bot.model_command("one", "/模型 model-b", "owner"))
        self.assertIn("已保存", await self.bot.chatgpt_thinking_command("one", "/思考 低", "owner"))
        self.assertEqual(self.bot.model_settings("private:self")["model"], "model-a")
        self.assertIsNone(self.bot.chatgpt_effort("private:self"))
        again = Bot(self.config, self.path)
        try:
            self.assertEqual(again.model_settings("one"), {"provider": "chatgpt", "model": "model-b"})
            self.assertEqual(again.chatgpt_effort("one"), "low")
        finally:
            again.db.close()

    async def test_invalid_choices_and_failed_catalog_do_not_save_settings(self):
        self.assertIn("未更改", await self.bot.model_command("one", "/模型 missing", "owner"))
        for text in ("/思考 ultra", "/思考 关闭", "/思考 随便"):
            self.assertIn("未更改", await self.bot.chatgpt_thinking_command("one", text, "owner"))
        self.assertIsNone(self.bot.chatgpt_effort("one"))
        self.bot.chatgpt.models.side_effect = ChatGPTError("Unavailable")
        with self.assertRaises(ChatGPTError):
            await self.bot.model_command("one", "/模型 model-b", "owner")
        self.assertEqual(self.bot.model_settings("one")["model"], "model-a")

    async def test_model_change_resets_incompatible_effort_and_defaults_are_independent(self):
        await self.bot.chatgpt_thinking_command("one", "/思考 高", "owner")
        result = await self.bot.model_command("one", "/模型 model-b", "owner")
        self.assertIn("已恢复模型默认", result)
        self.assertIsNone(self.bot.chatgpt_effort("one"))
        await self.bot.chatgpt_thinking_command("one", "/思考 低", "owner")
        await self.bot.chatgpt_thinking_command("one", "/思考 默认", "owner")
        self.assertEqual(self.bot.model_settings("one")["model"], "model-b")
        await self.bot.model_command("one", "/模型 默认", "owner")
        self.assertEqual(self.bot.model_settings("one")["model"], "model-a")

    async def test_listing_uses_catalog_without_app_only_efforts(self):
        result = await self.bot.model_command("one", "/模型 列表", "")
        self.assertIn("model-a", result)
        self.assertIn("model-b", result)
        self.assertNotIn("ultra", result)
        self.assertIsNone(self.bot.db.execute("SELECT 1 FROM group_model").fetchone())

    async def test_each_request_uses_its_conversation_provider_and_effort(self):
        self.bot.chatgpt.answer = AsyncMock(return_value=("ChatGPT reply", {}))
        self.bot.timeout_command("chat", "/超时 240", "owner")
        await self.bot.chatgpt_thinking_command("chat", "/思考 高", "owner")
        await self.bot.answer([{"role": "user", "content": "hello"}], True, "chat")
        self.assertEqual(self.bot.chatgpt.answer.call_args.kwargs["reasoning_effort"], "high")
        self.assertEqual(self.bot.chatgpt.answer.call_args.kwargs["timeout_seconds"], 240)
        self.bot.model_catalog = AsyncMock(return_value=[{"slug": "deep-model", "efforts": ["low", "high", "max"]}])
        await self.bot.model_command("deep", "/模型 deepseek deep-model", "owner")
        self.bot.c["deepseek_key"] = "key"
        response = MagicMock()
        response.status = 200
        response.json = AsyncMock(return_value={"choices": [{"message": {"content": "DeepSeek reply"}, "finish_reason": "stop"}]})
        self.bot.http.post.return_value.__aenter__ = AsyncMock(return_value=response)
        self.bot.http.post.return_value.__aexit__ = AsyncMock(return_value=False)
        self.bot.thinking_command("deep", "/思考 高", "owner")
        self.assertEqual(await self.bot.answer([], True, "deep"), "DeepSeek reply")
        sent = self.bot.http.post.call_args.kwargs["json"]
        self.assertEqual(sent["model"], "deep-model")
        self.assertEqual(sent["reasoning_effort"], "high")
        self.assertEqual(self.bot.http.post.call_args.kwargs["timeout"].total, 90)
        self.assertEqual(self.bot.chatgpt.answer.await_count, 1)
        self.assertIn("DeepSeek", self.bot.status("deep"))
        self.assertIn("ChatGPT", self.bot.status("chat"))
