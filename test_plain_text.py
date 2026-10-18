import asyncio
import contextlib
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock

from bot import Bot, qq_plain_text
from test_personal import event


class PlainTextTests(unittest.TestCase):
    def test_prose_keeps_structure_without_markdown_marks(self):
        source = "## 建议\n\n**重点**：先 _保存_，再*重启*。\n- 第一步\n+ 第二步\n> 引用\n\n---\n~~过时~~"
        expected = "建议\n\n重点：先 保存，再重启。\n• 第一步\n• 第二步\n引用\n\n过时"
        self.assertEqual(qq_plain_text(source), expected)

    def test_code_keeps_operators_indentation_and_html(self):
        code = 'def __init__(self):\n    return 2 ** 3, "# heading", "<b>raw</b>"\n\n\n# comment'
        result = qq_plain_text("示例：\n```python\n" + code + "\n```\n行内 `user_name` 和 `a * b`。")
        self.assertEqual(result, "示例：\n" + code + "\n行内 user_name 和 a * b。")
        self.assertEqual(qq_plain_text("~~~python\nx = 2 ** 3\n"), "x = 2 ** 3")

    def test_table_becomes_readable_records(self):
        source = "| **服务** | 特点 |\n| :--- | ---: |\n| ChatGPT | 连续对话 |\n| DeepSeek | API 调用 |"
        self.assertEqual(qq_plain_text(source), "服务：ChatGPT；特点：连续对话\n服务：DeepSeek；特点：API 调用")

    def test_links_are_visible_and_ordinary_symbols_survive(self):
        source = "[文档](https://example.com/guide)\n<https://example.com/>\nuser_name，2 * 3，a < b > c，1、保存。"
        self.assertEqual(qq_plain_text(source), "文档（https://example.com/guide）\nhttps://example.com/\nuser_name，2 * 3，a < b > c，1、保存。")

    def test_qq_control_tokens_are_removed(self):
        self.assertEqual(qq_plain_text("你好 <@123> [CQ:at,qq=456]"), "你好")


class PlainTextIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_group_and_private_send_and_remember_plain_text(self):
        bot = Bot({"owner_only": False}, ":memory:")
        bot.answer = AsyncMock(return_value="## 回答\n**收到**，用 `user_name`。")
        bot.qq = AsyncMock()
        worker = asyncio.create_task(bot.worker())
        try:
            for kind, mid in (("GROUP_AT_MESSAGE_CREATE", "group"), ("C2C_MESSAGE_CREATE", "private")):
                bot.accept(kind, event(mid, "hello"))
            await asyncio.wait_for(bot.queue.join(), 2)
            self.assertEqual(bot.qq.await_count, 2)
            for call in bot.qq.call_args_list:
                self.assertEqual(call.args[2]["msg_type"], 0)
                self.assertEqual(call.args[2]["content"], "回答\n收到，用 user_name。")
            for group in ("group", "private:self"):
                self.assertEqual(bot.history[group][-1]["content"], "回答\n收到，用 user_name。")
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            bot.db.close()

    async def test_both_providers_receive_plain_text_rule_with_custom_prompt(self):
        data_dir = tempfile.TemporaryDirectory()
        self.addCleanup(data_dir.cleanup)
        for provider in ("chatgpt", "deepseek"):
            with self.subTest(provider=provider):
                bot = Bot({"provider": provider, "system_prompt": "自定义风格", "deepseek_key": "test",
                           "data_dir": data_dir.name}, ":memory:")
                try:
                    if provider == "chatgpt":
                        bot.chatgpt.answer = AsyncMock(return_value=("收到", {}))
                    else:
                        response = MagicMock(status=200)
                        response.json = AsyncMock(return_value={"choices": [{"message": {"content": "收到"}}]})
                        bot.http = MagicMock()
                        bot.http.post.return_value.__aenter__ = AsyncMock(return_value=response)
                    await bot.answer([], True, "test")
                    prompt = (bot.chatgpt.answer.call_args.args[2] if provider == "chatgpt"
                              else bot.http.post.call_args.kwargs["json"]["messages"][0]["content"])
                    self.assertTrue(prompt.startswith("自定义风格"))
                    self.assertIn("QQ 纯文本", prompt)
                finally:
                    bot.db.close()
