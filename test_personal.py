import asyncio
import contextlib
import datetime as dt
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock

from bot import Bot, load_config


def event(mid, content, user="self", group="group"):
    return {"id": mid, "content": content, "group_openid": group,
            "author": {"member_openid": user, "user_openid": user, "username": "User"},
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()}


class PersonalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = Bot({"binding_code": "one-time-code"}, ":memory:")
        self.bot.answer = AsyncMock(return_value="模型回答")
        self.bot.qq = AsyncMock(return_value={"id": "reply"})

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

    async def test_unbound_and_other_users_never_enter_memory_or_call_model(self):
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", event("before", "hello"))
        self.assertTrue(self.bot.queue.empty())
        self.assertFalse(self.bot.history)
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", event("bind", "/绑定 one-time-code"))
        await self.drain()
        self.bot.answer.assert_not_awaited()
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", event("other", "secret", user="other"))
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", event("other-command", "/清空", user="other"))
        await self.drain()
        self.bot.answer.assert_not_awaited()
        self.assertFalse(self.bot.history["group"])
        self.assertEqual(self.bot.qq.await_count, 1)
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", event("self", "hello"))
        await self.drain()
        self.bot.answer.assert_awaited_once()

    async def test_private_chat_replies_to_user_and_uses_separate_memory(self):
        self.bot.accept("C2C_MESSAGE_CREATE", event("bind-private", "/绑定 one-time-code"))
        await self.drain()
        self.bot.accept("C2C_MESSAGE_CREATE", event("private", "private topic"))
        await self.drain()
        self.assertEqual(self.bot.qq.call_args.args[1], "/v2/users/self/messages")
        self.assertIn("private topic", self.bot.history["private:self"][0]["content"])
        self.bot.db.execute("INSERT INTO owners VALUES ('group:group', 'self')")
        self.bot.db.commit()
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", event("group", "group topic"))
        await self.drain()
        request_history = self.bot.answer.call_args.args[0]
        self.assertTrue(all("private topic" not in item["content"] for item in request_history))
        self.assertEqual(self.bot.qq.call_args.args[1], "/v2/groups/group/messages")

    async def test_wrong_expired_and_replayed_binding_do_not_replace_owner(self):
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", event("wrong", "/绑定 wrong", user="other"))
        self.assertEqual(self.bot.db.execute("SELECT COUNT(*) FROM owners").fetchone()[0], 0)
        self.bot.binding_until = time.time() - 1
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", event("expired", "/绑定 one-time-code", user="other"))
        self.assertEqual(self.bot.db.execute("SELECT COUNT(*) FROM owners").fetchone()[0], 0)
        self.bot.binding_until = time.time() + 600
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", event("good", "/绑定 one-time-code"))
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", event("replay", "/绑定 one-time-code", user="other"))
        self.assertEqual(self.bot.db.execute("SELECT user_id FROM owners").fetchone()[0], "self")

    async def test_owner_can_clear_memory_without_qq_admin_role(self):
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", event("bind", "/绑定 one-time-code"))
        await self.drain()
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", event("question", "hello"))
        await self.drain()
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", event("clear", "/清空"))
        await self.drain()
        self.assertFalse(self.bot.history["group"])
        self.assertIsNone(self.bot.db.execute("SELECT 1 FROM group_memory WHERE grp='group'").fetchone())

    async def test_binding_persists_and_default_auto_reply_is_off(self):
        self.assertFalse(self.bot.frequency("group")["auto_chat"])
        with tempfile.TemporaryDirectory() as folder:
            first = Bot({"binding_code": "code"}, folder + "/state.db")
            first.accept("GROUP_AT_MESSAGE_CREATE", event("bind", "/绑定 code"))
            first.db.close()
            second = Bot({}, folder + "/state.db")
            try:
                second.accept("GROUP_AT_MESSAGE_CREATE", event("after-restart", "hello"))
                self.assertEqual(second.queue.qsize(), 1)
            finally:
                second.db.close()


class ConfigTests(unittest.TestCase):
    def test_provider_override_uses_separate_model_and_relative_key_path(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            key = Path(folder) / "deepseek.key"
            key.write_text("test-key", encoding="utf-8")
            conf = {"app_id": "test-app", "app_secret": "test-secret", "provider": "chatgpt",
                    "chatgpt_model": "chat-model", "deepseek_model": "deep-model", "deepseek_key_file": "deepseek.key"}
            path.write_text(json.dumps(conf), encoding="utf-8")
            chosen = load_config(path, "deepseek")
            self.assertEqual(chosen["provider"], "deepseek")
            self.assertEqual(chosen["model"], "deep-model")
            self.assertEqual(chosen["deepseek_key"], "test-key")
            self.assertEqual(chosen["data_dir"], str(Path(folder) / ".data"))
            key.unlink()
            normal = load_config(path)
            self.assertEqual(normal["model"], "chat-model")
            self.assertNotIn("deepseek_key", normal)
            self.assertEqual(json.loads(path.read_text())["provider"], "chatgpt")

    def test_legacy_model_is_not_carried_to_a_different_provider(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"app_id": "app", "app_secret": "secret", "provider": "deepseek", "model": "old-model"}))
            self.assertEqual(load_config(path, "chatgpt")["model"], "")
            with self.assertRaises(ValueError):
                load_config(path, "unknown")
