import asyncio
import contextlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from bot import ApiError, Bot, load_config
from test_bot import message
from test_personal import event


class TimeoutSettingsTests(unittest.TestCase):
    def test_settings_persist_are_isolated_and_validate_permissions_and_range(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "state.db")
            config = {"provider": "chatgpt", "data_dir": folder, "model_timeout_seconds": 180}
            bot = Bot(config, path)
            try:
                self.assertIn("绑定账号", bot.timeout_command("group", "/超时 240", ""))
                self.assertIn("已保存", bot.timeout_command("group", "/等待 240", "owner"))
                for value in ("0", "29", "241", "300", "-1", "60.5", "True", "60 extra", "9" * 5000):
                    self.assertIn("未更改", bot.timeout_command("group", "/超时 " + value, "owner"))
                self.assertEqual(bot.model_timeout("group"), 240)
                self.assertEqual(bot.model_timeout("private:self"), 180)
                again = Bot(config, path)
                try:
                    self.assertEqual(again.model_timeout("group"), 240)
                    self.assertIn("240 秒", again.status("group"))
                    again.timeout_command("group", "/超时 默认", "owner")
                    self.assertEqual(again.model_timeout("group"), 180)
                finally:
                    again.db.close()
            finally:
                bot.db.close()

    def test_config_validation_and_legacy_deepseek_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            for value in (True, "150", 29, 241, 60.5):
                path.write_text(json.dumps({"model_timeout_seconds": value}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(path, require_qq=False)
        bot = Bot({}, ":memory:")
        try:
            self.assertEqual(bot.model_timeout("group"), 55)
            bot.c["thinking_mode"] = "at"
            self.assertEqual(bot.model_timeout("group"), 90)
            self.assertEqual(bot.model_timeout("group", False), 55)
        finally:
            bot.db.close()


class TimeoutWorkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = Bot({"owner_only": False}, ":memory:")
        self.bot.qq = AsyncMock()
        self.bot.answer = AsyncMock(return_value="完整回答")

    def tearDown(self):
        self.bot.db.close()

    async def drain(self):
        worker = asyncio.create_task(self.bot.worker())
        try:
            await asyncio.wait_for(self.bot.queue.join(), 2)
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker

    def replies(self):
        return [call.args[2] for call in self.bot.qq.call_args_list]

    async def test_fast_answer_has_no_progress_message(self):
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message())
        await self.drain()
        self.assertEqual([(r["content"], r["msg_seq"]) for r in self.replies()], [("完整回答", 1)])

    async def test_progress_then_complete_reply_use_distinct_sequences_and_only_final_memory(self):
        released = asyncio.Event()

        async def slow_answer(*args, **kwargs):
            await released.wait()
            return "完整回答"

        async def send(*args):
            if args[2]["content"] == "正在思考，请稍等。":
                released.set()

        self.bot.answer.side_effect = slow_answer
        self.bot.qq.side_effect = send
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message())
        with patch("bot.MODEL_PROGRESS_SECONDS", 0.01):
            await self.drain()
        self.assertEqual([(r["content"], r["msg_seq"]) for r in self.replies()],
                         [("正在思考，请稍等。", 1), ("完整回答", 2)])
        self.assertEqual({r["msg_id"] for r in self.replies()}, {"m1"})
        self.assertEqual([m["content"] for m in self.bot.history["g1"] if m["role"] == "assistant"], ["完整回答"])

    async def test_deadline_cancels_request_after_progress_and_worker_handles_next_job(self):
        cancelled = asyncio.Event()

        async def blocked(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        self.bot.answer.side_effect = blocked
        self.bot.model_timeout = MagicMock(return_value=0.06)
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message())
        with patch("bot.MODEL_PROGRESS_SECONDS", 0.01):
            await self.drain()
        self.assertTrue(cancelled.is_set())
        self.assertEqual([r["msg_seq"] for r in self.replies()], [1, 2])
        self.assertIn("超时", self.replies()[-1]["content"])
        self.bot.answer.side_effect = None
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message("next"))
        await self.drain()
        self.assertEqual(self.replies()[-1]["content"], "完整回答")

    async def test_failed_progress_delivery_does_not_lose_completed_answer(self):
        released = asyncio.Event()

        async def slow_answer(*args, **kwargs):
            await released.wait()
            return "完整回答"

        async def send(*args):
            if args[2]["msg_seq"] == 1:
                released.set()
                raise ApiError("delivery unknown")

        self.bot.answer.side_effect = slow_answer
        self.bot.qq.side_effect = send
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message())
        with patch("bot.MODEL_PROGRESS_SECONDS", 0.01):
            await self.drain()
        self.assertEqual(self.replies()[-1]["content"], "完整回答")
        self.assertEqual(self.replies()[-1]["msg_seq"], 2)

    async def test_private_progress_and_deadline_do_not_reset_while_sending(self):
        cancelled = asyncio.Event()

        async def blocked(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def send(*args):
            if args[2]["msg_seq"] == 1:
                await asyncio.wait_for(cancelled.wait(), 1)

        self.bot.answer.side_effect = blocked
        self.bot.qq.side_effect = send
        self.bot.model_timeout = MagicMock(return_value=0.06)
        self.bot.accept("C2C_MESSAGE_CREATE", event("private", "hello"))
        with patch("bot.MODEL_PROGRESS_SECONDS", 0.01):
            await self.drain()
        self.assertTrue(cancelled.is_set())
        self.assertEqual([r["msg_seq"] for r in self.replies()], [1, 2])
        self.assertIn("超时", self.replies()[-1]["content"])
        self.assertTrue(all(call.args[1] == "/v2/users/self/messages" for call in self.bot.qq.call_args_list))

    async def test_automatic_chat_does_not_emit_progress_or_timeout_reply(self):
        async def blocked(*args, **kwargs):
            await asyncio.Event().wait()

        self.bot.c.update(auto_chat=True, auto_min_messages=1)
        self.bot.answer.side_effect = blocked
        self.bot.model_timeout = MagicMock(return_value=0.04)
        self.bot.accept("GROUP_MESSAGE_CREATE", message())
        with patch("bot.MODEL_PROGRESS_SECONDS", 0.01):
            await self.drain()
        self.bot.answer.assert_awaited_once()
        self.bot.qq.assert_not_awaited()

    async def test_timeout_commands_route_without_model_or_history_and_members_cannot_change(self):
        self.bot.c.update(owner_only=True, group_members_can_chat=True)
        self.bot.db.execute("INSERT INTO owners VALUES ('group:group', 'self')")
        self.bot.db.commit()
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", event("set", "/等待 240"))
        await self.drain()
        self.assertEqual(self.bot.model_timeout("group"), 240)
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", event("denied", "/超时 30", user="other"))
        await self.drain()
        self.assertIn("绑定账号", self.replies()[-1]["content"])
        self.assertEqual(self.bot.model_timeout("group"), 240)
        self.bot.answer.assert_not_awaited()
        self.assertFalse(self.bot.history["group"])

    async def test_queue_wait_is_subtracted_from_reply_window(self):
        cancelled = asyncio.Event()

        async def blocked(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        self.bot.answer.side_effect = blocked
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message())
        priority, serial, job = self.bot.queue.get_nowait()
        self.bot.queue.task_done()
        job = list(job)
        job[4] = time.time() - 269.95
        self.bot.queue.put_nowait((priority, serial, tuple(job)))
        await self.drain()
        self.assertTrue(cancelled.is_set())
        self.assertEqual(len(self.replies()), 1)
        self.assertIn("超时", self.replies()[0]["content"])

    async def test_worker_shutdown_cancels_inflight_generation(self):
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def blocked(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        self.bot.answer.side_effect = blocked
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message())
        worker = asyncio.create_task(self.bot.worker())
        await asyncio.wait_for(started.wait(), 1)
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker
        self.assertTrue(cancelled.is_set())
        self.bot.qq.assert_not_awaited()
