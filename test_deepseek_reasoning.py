import asyncio
import contextlib
import json
import unittest
from unittest.mock import AsyncMock, patch

from bot import ApiError, Bot, DeepSeekAnswer, reasoning_messages
from test_bot import message
from test_personal import event


class ReasoningMessageTests(unittest.TestCase):
    def test_long_reasoning_is_split_and_bounded_without_losing_short_input(self):
        text = "思考内容。" * 700
        parts = reasoning_messages(text)
        self.assertEqual(len(parts), 2)
        self.assertEqual("".join(part.split("\n", 1)[1] for part in parts), text)
        parts = reasoning_messages(text * 5)
        self.assertEqual(len(parts), 3)
        self.assertTrue(all(len(part) <= 3000 for part in parts))
        self.assertIn("后续部分已省略", parts[-1])
        self.assertEqual(reasoning_messages("  "), [])


class ReasoningWorkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = Bot({"owner_only": False}, ":memory:")
        self.bot.qq = AsyncMock()
        self.bot.answer = AsyncMock(return_value=DeepSeekAnswer("最终回答", "这是一段思考过程"))

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

    async def test_reasoning_precedes_answer_and_history_contains_only_answer(self):
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message())
        await self.drain()
        replies = self.replies()
        self.assertEqual([r["msg_seq"] for r in replies], [1, 2])
        self.assertIn("这是一段思考过程", replies[0]["content"])
        self.assertEqual(replies[1]["content"], "DeepSeek 回答：\n最终回答")
        self.assertEqual(self.bot.history["g1"][-1]["content"], "最终回答")
        saved = self.bot.db.execute("SELECT messages FROM group_memory WHERE grp='g1'").fetchone()[0]
        self.assertNotIn("思考过程", saved)
        self.bot.answer.return_value = "后续普通回答"
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message("second"))
        await self.drain()
        self.assertEqual(len(self.replies()), 3)
        self.assertNotIn("思考过程", json.dumps(self.bot.answer.call_args.args[0], ensure_ascii=False))

    async def test_no_reasoning_produces_only_final_answer(self):
        self.bot.answer.return_value = DeepSeekAnswer("最终回答", "")
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message())
        await self.drain()
        self.assertEqual(len(self.replies()), 1)
        self.assertEqual(self.replies()[0]["content"], "最终回答")

    async def test_progress_plus_three_reasoning_parts_plus_answer_fit_five_replies(self):
        released = asyncio.Event()

        async def slow_answer(*args, **kwargs):
            await released.wait()
            return DeepSeekAnswer("最终回答", "思考" * 8000)

        async def send(*args):
            if args[2]["content"] == "正在思考，请稍等。":
                released.set()

        self.bot.answer.side_effect = slow_answer
        self.bot.qq.side_effect = send
        self.bot.accept("C2C_MESSAGE_CREATE", event("private", "hello"))
        with patch("bot.MODEL_PROGRESS_SECONDS", 0.01):
            await self.drain()
        replies = self.replies()
        self.assertEqual([r["msg_seq"] for r in replies], [1, 2, 3, 4, 5])
        self.assertTrue(all(len(r["content"]) <= 3000 for r in replies))
        self.assertEqual({r["msg_id"] for r in replies}, {"private"})
        self.assertIn("后续部分已省略", replies[3]["content"])
        self.assertIn("最终回答", replies[-1]["content"])
        self.assertTrue(all(call.args[1] == "/v2/users/self/messages" for call in self.bot.qq.call_args_list))

    async def test_failed_reasoning_is_diagnosed_and_final_answer_still_attempted(self):
        async def send(*args):
            if args[2]["msg_seq"] == 1:
                raise ApiError("HTTP=400 code=123", status=400, code=123, response={"message": "rejected"})

        self.bot.qq.side_effect = send
        self.bot.answer.return_value = DeepSeekAnswer("最终回答", "思考" * 4000)
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message())
        await self.drain()
        replies = self.replies()
        self.assertEqual([r["msg_seq"] for r in replies], [1, 2])
        self.assertIn("部分思考过程未能发送", replies[-1]["content"])
        self.assertIn("最终回答", replies[-1]["content"])
        stage, answer, details = self.bot.db.execute("SELECT stage,answer,details FROM send_failures").fetchone()
        self.assertEqual(stage, "reasoning")
        self.assertEqual(answer, "思考" * 4000)
        self.assertEqual(json.loads(details)["code"], 123)
        self.assertEqual(self.bot.history["g1"][-1]["content"], "最终回答")

    async def test_auto_skip_does_not_publish_reasoning(self):
        self.bot.c.update(auto_chat=True, auto_min_messages=1)
        self.bot.answer.return_value = DeepSeekAnswer("[SKIP]", "选择保持安静的过程")
        self.bot.accept("GROUP_MESSAGE_CREATE", message())
        await self.drain()
        self.bot.qq.assert_not_awaited()
