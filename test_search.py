import asyncio
import contextlib
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock

from bot import Bot, qq_reply, wants_sources
from chatgpt import ChatGPT, ChatGPTError, answer_with_citations, collect_response
from test_chatgpt import Reply, sse
from test_personal import event


SOURCE = {"title": "Python 官网", "url": "https://www.python.org/downloads/"}


def cited_message():
    citation = "([Python 官网](https://www.python.org/downloads/))"
    text = "已发布新版本。" + citation
    return {"id": "message", "type": "message", "role": "assistant", "phase": "final_answer",
            "content": [{"type": "output_text", "text": text, "annotations": [
                {"type": "url_citation", "start_index": len("已发布新版本。"), "end_index": len(text), **SOURCE}]}]}


class SearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_stream_items_supply_answer_sources_and_tool_count(self):
        events = [{"type": "response.output_item.done", "item": {"type": "reasoning", "id": "private"}},
                  {"type": "response.output_item.done", "item": {"type": "web_search_call", "id": "s1", "status": "completed"}},
                  {"type": "response.output_item.done", "item": {"type": "web_search_call", "id": "s2", "status": "completed"}},
                  {"type": "response.output_item.done", "item": cited_message()},
                  {"type": "response.completed", "response": {"status": "completed", "output": [], "usage": {"input_tokens": 42}}}]
        text, usage = await collect_response(sse(events))
        self.assertEqual(text, "已发布新版本。〔1〕")
        self.assertEqual(usage["web_search_calls"], 2)
        self.assertEqual(usage["web_sources"], [SOURCE])
        self.assertEqual(usage["input_tokens"], 42)

    async def test_final_response_output_and_duplicate_urls_keep_one_source(self):
        message = cited_message()
        message["content"] *= 2
        text, usage = await collect_response(sse([{"type": "response.completed", "response": {"output": [message]}}]))
        self.assertEqual(text.count("〔1〕"), 2)
        self.assertEqual(usage["web_sources"], [SOURCE])

    async def test_failed_stream_does_not_publish_search_result_or_partial_answer(self):
        events = [{"type": "response.output_item.done", "item": cited_message()},
                  {"type": "response.failed", "response": {"error": {"code": "subscription_sharing_usage_limit_exceeded"}}}]
        with self.assertRaises(ChatGPTError):
            await collect_response(sse(events))

    async def test_search_is_explicitly_enabled_without_changing_other_request_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            client = ChatGPT(directory)
            client.resolve_model = AsyncMock(return_value={"slug": "test-model", "supported_reasoning_levels": [{"effort": "high"}]})
            client.access_token = AsyncMock(return_value="test-token")
            session = MagicMock()
            for enabled in (True, False):
                response = Reply({})
                response.content = sse([{"type": "response.completed", "response": {"output": [cited_message()]}}])
                session.post.return_value = response
                await client.answer(session, [], "help", reasoning_effort="high", web_search=enabled)
                body = session.post.call_args.kwargs["json"]
                self.assertEqual(body["store"], False)
                self.assertEqual(body["stream"], True)
                self.assertEqual(body["reasoning"], {"effort": "high"})
                if enabled:
                    self.assertEqual(body["tools"], [{"type": "web_search"}])
                    self.assertEqual(body["tool_choice"], "auto")
                else:
                    self.assertNotIn("tools", body)

    def test_citation_offsets_never_remove_claims_and_invalid_urls_are_ignored(self):
        message = cited_message()
        part = message["content"][0]
        part["annotations"][0].update(start_index=0, end_index=7)
        part["annotations"].append({"type": "url_citation", "url": "javascript:alert(1)"})
        text, sources = answer_with_citations([message])
        self.assertIn("已发布新版本。", text)
        self.assertEqual(sources, [SOURCE])

    def test_qq_hides_sources_by_default_and_shows_complete_urls_on_request(self):
        self.assertEqual(qq_reply("**结果**。〔1〕", [SOURCE]), "结果。")
        shown = qq_reply("**结果**。〔1〕", [SOURCE], show_sources=True)
        self.assertIn("结果。〔1〕", shown)
        self.assertIn(SOURCE["url"], shown)
        self.assertNotIn("**", shown)
        long_reply = qq_reply("答" * 4000 + "〔1〕", [SOURCE], show_sources=True)
        self.assertLessEqual(len(long_reply), 3000)
        self.assertTrue(long_reply.endswith(SOURCE["url"]))
        self.assertTrue(wants_sources("再给我来源网址"))
        self.assertFalse(wants_sources("不用显示网页来源，查完直接说结果"))

    async def test_bot_uses_search_by_default_and_status_reports_actual_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = Bot({"provider": "chatgpt", "data_dir": directory, "owner_only": False}, ":memory:")
            bot.chatgpt.answer = AsyncMock(return_value=("新版本。〔1〕", {"web_search_calls": 2, "web_sources": [SOURCE]}))
            bot.qq = AsyncMock()
            worker = asyncio.create_task(bot.worker())
            try:
                bot.accept("GROUP_AT_MESSAGE_CREATE", event("group", "联网查最新版本"))
                await asyncio.wait_for(bot.queue.join(), 2)
                self.assertEqual(bot.qq.call_args.args[2]["content"], "新版本。")
                self.assertTrue(bot.chatgpt.answer.call_args.kwargs["web_search"])
                self.assertIn("检索/读取网页2次", bot.status("group"))
                self.assertEqual(bot.history["group"][-1]["content"], "新版本。")
                bot.accept("C2C_MESSAGE_CREATE", event("private", "联网查询，附上来源网址"))
                await asyncio.wait_for(bot.queue.join(), 2)
                self.assertIn(SOURCE["url"], bot.qq.call_args.args[2]["content"])
                bot.c["chatgpt_web_search"] = False
                await bot.answer([], True, "group")
                self.assertFalse(bot.chatgpt.answer.call_args.kwargs["web_search"])
                self.assertIn("网页搜索：已关闭", bot.status("group"))
            finally:
                worker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker
                bot.db.close()
