import asyncio
import contextlib
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

from bot import load_config
from weixin import (API_BASE, LoginExpired, WeixinAPI, WeixinBot, WeixinError,
                    account_name, api_base, credentials, login, private_message, read_json, save_account, serve)


def record(user="self", bot="bot", generation="generation"):
    return {"bot_token": "test-token", "user_id": user, "bot_id": bot,
            "base_url": API_BASE, "login_id": generation}


def message(mid="1", text="你好", **extra):
    return {"message_id": mid, "from_user_id": "self", "to_user_id": "bot",
            "message_type": 1, "message_state": 2, "create_time_ms": int(time.time() * 1000),
            "context_token": "reply-context", "item_list": [{"type": 1, "text_item": {"text": text}}], **extra}


class WeixinTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.config = {"provider": "deepseek", "data_dir": str(self.folder)}
        self.bot = WeixinBot(self.config, self.folder, record())
        self.bot.answer = AsyncMock(return_value="回答")
        self.api = AsyncMock()

    def tearDown(self):
        self.bot.db.close()
        self.temp.cleanup()

    def ingest(self, *messages, cursor="cursor"):
        self.bot.ingest({"msgs": list(messages), "get_updates_buf": cursor})

    async def test_only_owner_private_text_reaches_model(self):
        for mid, extra in enumerate(({"from_user_id": "other"}, {"group_id": "group"},
                                    {"chat_type": "group"}, {"to_user_id": "other-bot"},
                                    {"message_type": 2}, {"message_state": 1},
                                    {"context_token": ""}, {"delete_time_ms": 1},
                                    {"create_time_ms": (time.time() - 3700) * 1000})):
            self.ingest(message(str(mid), "不能泄露", **extra))
        self.assertEqual(self.bot.db.execute("SELECT COUNT(*) FROM wx_inbox").fetchone()[0], 0)
        await self.bot.process_pending(self.api)
        self.bot.answer.assert_not_awaited()
        self.assertFalse(self.bot.history)
        self.api.send.assert_not_awaited()
        self.ingest(message("good"))
        await self.bot.process_pending(self.api)
        self.bot.answer.assert_awaited_once()
        self.api.send.assert_awaited_once()
        self.assertEqual(self.bot.history[self.bot.scope][-1]["content"], "回答")

    async def test_replay_and_uint64_ids_do_not_duplicate_model_or_reply(self):
        msg = message(18446744073709551615)
        self.ingest(msg, copy.deepcopy(msg))
        await self.bot.process_pending(self.api)
        self.ingest(msg)
        await self.bot.process_pending(self.api)
        self.bot.answer.assert_awaited_once()
        self.api.send.assert_awaited_once()
        row = self.bot.db.execute("SELECT id,payload,reply,done FROM wx_inbox").fetchone()
        self.assertEqual(row, ("18446744073709551615", "", None, 1))

    async def test_failed_send_survives_restart_with_same_reply_and_client_id(self):
        self.ingest(message())
        self.api.send.side_effect = asyncio.TimeoutError
        with self.assertRaises(asyncio.TimeoutError):
            await self.bot.process_pending(self.api)
        first_id = self.api.send.call_args.args[2]
        self.bot.answer.assert_awaited_once()
        self.bot.db.close()
        self.bot = WeixinBot(self.config, self.folder, record())
        self.bot.answer = AsyncMock(return_value="不应重新生成")
        self.api.send.reset_mock(side_effect=True)
        await self.bot.process_pending(self.api)
        self.bot.answer.assert_not_awaited()
        self.assertEqual(self.api.send.call_args.args[1:], ("回答", first_id))
        self.assertEqual(len(self.bot.history[self.bot.scope]), 2)
        self.assertEqual(self.bot.db.execute("SELECT cursor FROM wx_cursor").fetchone()[0], "cursor")

    async def test_followups_include_previous_answer_and_commands_do_not_call_model(self):
        self.ingest(message("1", "第一个问题"), message("2", "继续"), message("3", "/记忆 200"))
        await self.bot.process_pending(self.api)
        self.assertEqual(self.bot.answer.await_count, 2)
        history = self.bot.answer.call_args.args[0]
        self.assertEqual([m["content"] for m in history], ["第一个问题", "回答", "继续"])
        self.assertEqual(self.bot.memory_settings(self.bot.scope)["limit"], 200)
        self.ingest(message("4", "/清空"), message("5", "/帮助"))
        await self.bot.process_pending(self.api)
        self.assertFalse(self.bot.history[self.bot.scope])
        self.assertEqual(self.bot.memory_settings(self.bot.scope)["limit"], 200)
        self.assertEqual(self.bot.answer.await_count, 2)
        self.assertIn("各微信号的记忆和设置独立", self.api.send.call_args.args[1])

    async def test_two_accounts_have_independent_memory_settings_and_credentials(self):
        second_folder = self.folder / "second"
        second_folder.mkdir()
        second = WeixinBot(self.config, second_folder, record("second-user", "second-bot"))
        second.answer = AsyncMock(return_value="第二号回答")
        try:
            self.ingest(message("1", "第一个号的秘密"), message("2", "/记忆 200"))
            await self.bot.process_pending(self.api)
            second.ingest({"msgs": [message("1", "第二个号的问题", from_user_id="second-user", to_user_id="second-bot")],
                           "get_updates_buf": "second-cursor"})
            await second.process_pending(self.api)
            self.assertNotEqual(self.bot.scope, second.scope)
            self.assertEqual(second.answer.call_args.args[0], [{"role": "user", "content": "第二个号的问题"}])
            self.assertEqual(second.memory_settings(second.scope)["limit"], 100)
            self.assertEqual(self.bot.db.execute("SELECT cursor FROM wx_cursor").fetchone()[0], "cursor")
        finally:
            second.db.close()

    async def test_accounts_share_only_model_refresh_lock_and_use_weixin_prompt(self):
        from chatgpt import ChatGPT
        shared = ChatGPT(self.folder)
        one_dir, two_dir = self.folder / "one", self.folder / "two"
        one_dir.mkdir()
        two_dir.mkdir()
        one = WeixinBot(self.config, one_dir, record(), shared)
        two = WeixinBot(self.config, two_dir, record("other", "other-bot"), shared)
        try:
            self.assertIsNot(one.chatgpt, two.chatgpt)
            self.assertIs(one.chatgpt.refresh_lock, two.chatgpt.refresh_lock)
            self.assertNotIn("QQ", one.answer_prompt())
            self.assertIn("微信私聊", one.answer_prompt())
        finally:
            one.db.close()
            two.db.close()

    async def test_media_and_unknown_commands_do_not_call_model(self):
        self.ingest(message("image", item_list=[{"type": 2, "image_item": {"url": "private"}}]),
                    message("mixed", item_list=[{"type": 1, "text_item": {"text": "描述图片"}}, {"type": 2}]),
                    message("cmd", "/频率 开启"))
        await self.bot.process_pending(self.api)
        self.bot.answer.assert_not_awaited()
        self.assertEqual(self.api.send.await_count, 3)
        self.assertFalse(self.bot.history[self.bot.scope])
        self.assertFalse(self.bot.c["auto_chat"])

    async def test_new_login_discards_old_pending_but_preserves_same_owner_memory(self):
        self.ingest(message())
        self.api.send.side_effect = asyncio.TimeoutError
        with self.assertRaises(asyncio.TimeoutError):
            await self.bot.process_pending(self.api)
        self.bot.db.close()
        self.bot = WeixinBot(self.config, self.folder, record(generation="new-login"))
        self.assertTrue(self.bot.history[self.bot.scope])
        self.assertEqual(self.bot.db.execute("SELECT COUNT(*) FROM wx_inbox").fetchone()[0], 0)
        self.assertEqual(self.bot.db.execute("SELECT COUNT(*) FROM wx_cursor").fetchone()[0], 0)
        self.bot.db.close()
        self.bot = WeixinBot(self.config, self.folder, record("other-user", "other-bot", "third-login"))
        self.assertFalse(self.bot.history[self.bot.scope])

    async def test_expired_login_stops_polling_and_reports_local_status(self):
        with patch("weixin.WeixinAPI.updates", AsyncMock(side_effect=LoginExpired("expired"))) as updates:
            await self.bot.run_private()
            updates.assert_awaited_once()
        self.assertEqual(read_json(self.folder / "status.json")["state"], "login_required")

    async def test_login_verification_and_redirect_save_only_complete_credentials(self):
        responses = [
            {"qrcode": "opaque", "qrcode_img_content": "https://weixin.qq.com/qr"},
            {"status": "scaned_but_redirect", "redirect_host": "ilinkai.weixin.qq.com"},
            {"status": "need_verifycode"},
            {"status": "scaned"},
            {"status": "confirmed", "ilink_bot_id": "newbot", "ilink_user_id": "newuser", "bot_token": "newtoken",
             "baseurl": API_BASE},
        ]
        displayed = []
        with patch("weixin.WeixinAPI.request", AsyncMock(side_effect=responses)) as req, \
                patch("weixin.asyncio.sleep", AsyncMock()):
            await login(self.folder, "main", show_qr=displayed.append, verify_code=AsyncMock(return_value="123456"))
        saved = credentials(read_json(self.folder / "main" / "credentials.json"))
        self.assertEqual(saved["user_id"], "newuser")
        self.assertEqual(req.call_args_list[3].kwargs["params"]["verify_code"], "123456")
        self.assertEqual(len(displayed), 1)

    async def test_failed_login_keeps_existing_account(self):
        save_account(self.folder, "main", record())
        path = self.folder / "main" / "credentials.json"
        before = path.read_bytes()
        responses = [{"qrcode": "opaque", "qrcode_img_content": "qr"},
                     {"status": "confirmed", "ilink_bot_id": "bot", "bot_token": "new"}]
        with patch("weixin.WeixinAPI.request", AsyncMock(side_effect=responses)), self.assertRaises(WeixinError):
            await login(self.folder, "main", show_qr=lambda value: None)
        self.assertEqual(path.read_bytes(), before)

    async def test_supervisor_loads_new_account_without_restarting_first(self):
        save_account(self.folder, "main", record())
        started, ready = [], asyncio.Event()

        async def fake_run(bot):
            started.append(bot.alias)
            if bot.alias == "main":
                save_account(self.folder, "second", record("second-user", "second-bot"))
            if len(started) == 2:
                ready.set()
            await asyncio.Event().wait()

        real_sleep = asyncio.sleep
        with patch.object(WeixinBot, "run_private", fake_run), \
                patch("weixin.asyncio.sleep", lambda seconds: real_sleep(0.01)):
            task = asyncio.create_task(serve(self.config, self.folder))
            try:
                await asyncio.wait_for(ready.wait(), 2)
                self.assertEqual(started, ["main", "second"])
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task


class WeixinValidationTests(unittest.TestCase):
    def test_untrusted_hosts_and_alias_traversal_are_rejected(self):
        for url in ("http://ilinkai.weixin.qq.com", "https://weixin.qq.com.evil.test", "https://example.com",
                    "https://ilinkai.weixin.qq.com@evil.test", "https://ilinkai.weixin.qq.com:8443",
                    "https://ilinkai.weixin.qq.com/token", "https://ilinkai.weixin.qq.com?x=secret"):
            with self.subTest(url=url), self.assertRaises(WeixinError):
                api_base(url)
        for alias in ("../foo", "C:\\foo", "con", "nul", "com1", "", "MAIN"):
            with self.subTest(alias=alias), self.assertRaises(ValueError):
                account_name(alias)
        self.assertEqual(api_base(API_BASE + "/"), API_BASE)

    def test_weixin_config_needs_no_qq_credentials(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"provider": "chatgpt"}), encoding="utf-8")
            self.assertEqual(load_config(path, require_qq=False)["provider"], "chatgpt")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_duplicate_owner_cannot_be_saved_under_second_alias(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_account(root, "main", record())
            with self.assertRaises(WeixinError):
                save_account(root, "second", record(bot="another-bot"))
            self.assertFalse((root / "second" / "credentials.json").exists())


class FakeResponse:
    def __init__(self, data, status=200):
        self.data, self.status = data, status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def json(self, **kwargs):
        return self.data


class WeixinAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_accepts_real_json_response_with_weixin_octet_stream_type(self):
        import aiohttp
        from aiohttp import web

        async def handler(request):
            return web.Response(body=b'{"ret":0,"qrcode":"opaque","qrcode_img_content":"qr"}',
                                content_type="application/octet-stream")

        app = web.Application()
        app.router.add_post("/ilink/bot/get_bot_qrcode", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        try:
            async with aiohttp.ClientSession() as session:
                api = WeixinAPI(session)
                # Test transport only; production URL validation remains strict.
                api.base_url = "http://127.0.0.1:" + str(runner.addresses[0][1])
                self.assertEqual((await api.request("get_bot_qrcode", {}))["qrcode"], "opaque")
        finally:
            await runner.cleanup()

    async def test_wire_format_context_and_auth(self):
        from unittest.mock import Mock
        http = Mock()
        http.request.return_value = FakeResponse({"ret": 0})
        api = WeixinAPI(http, token="test-token")
        await api.send(message(run_id="run"), "回复", "stable-client-id")
        args, kwargs = http.request.call_args
        self.assertEqual(args, ("POST", API_BASE + "/ilink/bot/sendmessage"))
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-token")
        msg = kwargs["json"]["msg"]
        self.assertEqual(msg["context_token"], "reply-context")
        self.assertEqual(msg["to_user_id"], "self")
        self.assertEqual(msg["run_id"], "run")
        self.assertEqual(msg["client_id"], "stable-client-id")
        self.assertEqual(msg["item_list"], [{"type": 1, "text_item": {"text": "回复"}}])
        self.assertEqual(kwargs["json"]["base_info"]["bot_agent"], "QQBotGPT/1.0")

    async def test_error_body_is_never_exposed_and_long_poll_timeout_retains_cursor(self):
        from unittest.mock import Mock
        http = Mock()
        api = WeixinAPI(http)
        for body in ({"ret": -14, "errmsg": "secret"}, {"ret": 0, "errcode": -14, "errmsg": "secret"}):
            http.request.return_value = FakeResponse(body)
            with self.assertRaises(LoginExpired) as caught:
                await api.request("getupdates", {})
            self.assertNotIn("secret", str(caught.exception))
        http.request.return_value = FakeResponse({"ret": 42, "errmsg": "secret"})
        with self.assertRaises(WeixinError) as caught:
            await api.request("getupdates", {})
        self.assertNotIn("secret", str(caught.exception))
        with patch.object(api, "request", AsyncMock(side_effect=asyncio.TimeoutError)):
            self.assertEqual((await api.updates("cursor"))["get_updates_buf"], "cursor")


if __name__ == "__main__":
    unittest.main()
