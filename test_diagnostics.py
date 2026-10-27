import asyncio
import contextlib
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from bot import ApiError, Bot
from diagnostics import MAX_TEXT, RETENTION_SECONDS, SendFailureStore, read_failures
from test_bot import message


class SendFailureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "state.db"
        self.bot = Bot({"owner_only": False}, str(self.path))
        self.bot.answer = AsyncMock(return_value="**模型回答**")
        self.bot.access_token = AsyncMock(return_value="PRIVATE-TOKEN")
        self.bot.http = MagicMock()
        self.response = MagicMock()
        self.response.status = 400
        self.response.headers = {"X-Tps-Trace-ID": "trace-from-qq"}
        self.response.json = AsyncMock(return_value={"code": 40034006, "message": "example rejection", "data": {"reason": "detail"}})
        self.bot.http.request.return_value.__aenter__ = AsyncMock(return_value=self.response)
        self.bot.http.request.return_value.__aexit__ = AsyncMock(return_value=False)

    def tearDown(self):
        self.bot.db.close()
        self.temp.cleanup()

    async def drain(self):
        worker = asyncio.create_task(self.bot.worker())
        try:
            await asyncio.wait_for(self.bot.queue.join(), 2)
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker

    async def test_rejected_answer_saves_original_sent_text_and_qq_details_without_logging_content(self):
        original = "**模型回答**" + "这是回答内容。" * 600
        self.bot.answer.return_value = original
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message())
        with self.assertLogs("qqbot", "WARNING") as logs:
            await self.drain()
        rows = read_failures(self.path)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["answer"], original)
        self.assertEqual(row["content"], self.bot.http.request.call_args.kwargs["json"]["content"])
        self.assertLessEqual(len(row["content"]), 3000)
        self.assertEqual((row["conversation"], row["message_id"], row["stage"]), ("g1", "m1", "answer"))
        self.assertEqual(row["details"]["http_status"], 400)
        self.assertEqual(row["details"]["code"], 40034006)
        self.assertEqual(row["details"]["trace_id"], "trace-from-qq")
        self.assertEqual(row["details"]["response"]["message"], "example rejection")
        self.assertEqual(row["details"]["response"]["data"]["reason"], "detail")
        self.assertNotIn("PRIVATE-TOKEN", json.dumps(row))
        self.assertNotIn("模型回答", " ".join(logs.output))
        self.assertNotIn("example rejection", " ".join(logs.output))
        self.assertFalse(any(m["role"] == "assistant" for m in self.bot.history["g1"]))

    async def test_successful_send_is_not_recorded_as_failure(self):
        self.response.status = 200
        self.response.json.return_value = {"id": "ok"}
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", message())
        await self.drain()
        self.assertEqual(read_failures(self.path), [])
        self.assertEqual(self.bot.history["g1"][-1]["content"], "模型回答")

    async def test_non_json_error_keeps_bounded_body_and_trace_id(self):
        self.response.status = 502
        self.response.json.side_effect = ValueError("invalid JSON")
        self.response.text = AsyncMock(return_value="upstream unavailable " * 2000)
        with self.assertRaises(ApiError):
            await self.bot.send_reply("private:user", "m1", "answer", 2, stage="answer")
        row = read_failures(self.path)[0]
        self.assertEqual(row["details"]["http_status"], 502)
        self.assertTrue(row["details"]["response"]["truncated"])
        self.assertIn("upstream unavailable", row["details"]["response"]["raw_body"])
        self.assertEqual(row["details"]["trace_id"], "trace-from-qq")

    async def test_transport_error_keeps_answer_and_does_not_claim_qq_status(self):
        self.bot.http.request.side_effect = aiohttp.ClientConnectionError("connection closed")
        with self.assertRaises(aiohttp.ClientConnectionError):
            await self.bot.send_reply("g1", "m1", "answer", 1, stage="answer", answer="original")
        row = read_failures(self.path)[0]
        self.assertEqual(row["answer"], "original")
        self.assertEqual(row["details"]["error_type"], "ClientConnectionError")
        self.assertIsNone(row["details"]["http_status"])

    async def test_progress_timeout_is_recorded_and_final_answer_still_sent(self):
        async def never(*args, **kwargs):
            await asyncio.Event().wait()

        self.bot.qq = AsyncMock(side_effect=never)
        with self.assertRaises(asyncio.TimeoutError):
            await self.bot.send_reply("g1", "m1", "正在思考，请稍等。", 1, stage="progress", timeout=0.01)
        self.bot.qq.side_effect = None
        await self.bot.send_reply("g1", "m1", "complete", 2, stage="answer", answer="complete")
        rows = read_failures(self.path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["stage"], "progress")
        self.assertIsNone(rows[0]["answer"])

    async def test_storage_failure_does_not_replace_original_send_error(self):
        with patch.object(self.bot.send_failures, "record", side_effect=sqlite3.OperationalError("disk full")):
            with self.assertRaises(ApiError) as caught:
                await self.bot.send_reply("g1", "m1", "answer", 1, stage="answer")
        self.assertEqual(caught.exception.code, 40034006)

    async def test_explicit_clear_removes_only_current_conversation_diagnostics(self):
        for group in ("g1", "g2"):
            self.bot.send_failures.record(group, "m", "answer", "text", "text", {})
        self.response.status = 200
        self.response.json.return_value = {"id": "ok"}
        incoming = message(text="/清空")
        incoming["author"]["member_role"] = "admin"
        self.bot.accept("GROUP_AT_MESSAGE_CREATE", incoming)
        await self.drain()
        self.assertEqual([row["conversation"] for row in read_failures(self.path)], ["g2"])


class StoreTests(unittest.TestCase):
    def test_retention_count_limit_redaction_and_persistence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.db"
            conn = sqlite3.connect(path)
            store = SendFailureStore(conn)
            details = {"response": {"message": "rejected", "access_token": "private",
                                   "data": {"client_secret": "private", "text": "Bearer private app_secret=private"}}, "trace_id": "trace"}
            with patch("diagnostics.MAX_RECORDS", 3):
                for i in range(5):
                    store.record("g", str(i), "answer", "text", "original", details)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM send_failures").fetchone()[0], 3)
            conn.execute("UPDATE send_failures SET created=? WHERE message_id='2'", (time.time() - RETENTION_SECONDS - 5,))
            conn.commit()
            store.prune()
            conn.close()
            rows = read_failures(path)
            self.assertEqual([row["message_id"] for row in rows], ["4", "3"])
            self.assertNotIn("private", json.dumps(rows))
            self.assertEqual(rows[0]["details"]["response"]["message"], "rejected")
            self.assertEqual(rows[0]["details"]["trace_id"], "trace")

    def test_large_answers_and_error_bodies_are_bounded_with_truncation_markers(self):
        conn = sqlite3.connect(":memory:")
        try:
            store = SendFailureStore(conn)
            store.record("g", "m", "answer", "x" * (MAX_TEXT + 1), "y" * (MAX_TEXT + 1),
                         {"response": {str(i): "z" * 10000 for i in range(100)}})
            content, answer, raw = conn.execute("SELECT content,answer,details FROM send_failures").fetchone()
            self.assertEqual(len(content), MAX_TEXT)
            self.assertEqual(len(answer), MAX_TEXT)
            self.assertLessEqual(len(raw), MAX_TEXT)
            self.assertTrue(json.loads(raw)["answer_truncated"])
            self.assertTrue(json.loads(raw)["content_truncated"])
            self.assertTrue(json.loads(raw)["response"]["truncated"])
        finally:
            conn.close()

    def test_reader_is_read_only_and_handles_older_database(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.db"
            conn = sqlite3.connect(path)
            conn.execute("CREATE TABLE existing (id INTEGER)")
            conn.close()
            before = path.read_bytes()
            self.assertEqual(read_failures(path), [])
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(sqlite3.OperationalError):
                read_failures(Path(folder) / "missing.db")
            self.assertFalse((Path(folder) / "missing.db").exists())
