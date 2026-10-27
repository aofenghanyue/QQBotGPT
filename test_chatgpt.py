import asyncio
import json
import tempfile
import time
import unittest
from unittest.mock import MagicMock

from cryptography.hazmat.primitives.asymmetric import rsa
import jwt

from chatgpt import (ChatGPT, ChatGPTError, DIRECT_SCOPE, ISSUER, collect_response,
                     validate_callback, validate_identity, write_json)


async def sse(events):
    for event in events:
        yield ("data: " + json.dumps(event) + "\n").encode()
        yield b"\n"


class Reply:
    status = 200
    def __init__(self, data):
        self.data = data
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass
    async def json(self, **kwargs):
        await asyncio.sleep(0)
        return self.data


class IdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(cls.key.public_key()))
        cls.public["kid"] = "test"

    def token(self, **changes):
        claims = dict(iss=ISSUER, sub="account", aud="oaiapp_test", nonce="nonce",
                      iat=int(time.time()), exp=int(time.time()) + 600)
        claims.update(changes)
        return jwt.encode(claims, self.key, algorithm="RS256", headers={"kid": "test"})

    def test_identity_validates_signature_issuer_audience_nonce_expiry(self):
        keys = {"keys": [self.public]}
        self.assertEqual(validate_identity(self.token(), keys, "oaiapp_test", "nonce")["sub"], "account")
        for changes in ({"iss": "https://example.com"}, {"aud": "wrong"}, {"nonce": "wrong"}, {"exp": 1}):
            with self.subTest(changes=changes), self.assertRaises(ChatGPTError):
                validate_identity(self.token(**changes), keys, "oaiapp_test", "nonce")
        other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        keys["keys"][0] = {**json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(other.public_key())), "kid": "test"}
        with self.assertRaises(ChatGPTError):
            validate_identity(self.token(), keys, "oaiapp_test", "nonce")

    def test_returning_login_cannot_replace_another_account(self):
        with self.assertRaises(ChatGPTError):
            validate_identity(self.token(), {"keys": [self.public]}, "oaiapp_test", "nonce",
                              {"subject": "different", "issuer": ISSUER})

    def test_callback_state_and_issued_registration_are_required(self):
        query = {"state": "expected", "code": "code", "client_id": "oaiapp_test"}
        self.assertEqual(validate_callback(query, "expected", "dynamic_agent_client"), ("oaiapp_test", "code"))
        for bad in ({**query, "state": "wrong"}, {**query, "client_id": "dynamic_agent_client"},
                    {"state": "expected", "code": "code"}, {**query, "error": "access_denied"}):
            with self.subTest(bad=bad), self.assertRaises(ChatGPTError):
                validate_callback(bad, "expected", "dynamic_agent_client")
        self.assertEqual(validate_callback({"state": "expected", "code": "code"}, "expected", "saved"), ("saved", "code"))
        with self.assertRaises(ChatGPTError):
            validate_callback(query, "expected", "other")


class RequestTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.client = ChatGPT(self.temp.name)
        self.record = dict(client_id="saved-client", issuer=ISSUER, subject="account", email="test@example.invalid",
            access_token="old-access", refresh_token="old-refresh", scopes=[DIRECT_SCOPE], expires_at=1)
        write_json(self.client.path, {"active": "saved-client", "pending": None, "profiles": {"saved-client": self.record}})

    def tearDown(self):
        self.temp.cleanup()

    async def test_rotation_is_persistent_and_serialized(self):
        session = MagicMock()
        session.post.return_value = Reply(dict(access_token="new-access", refresh_token="new-refresh",
                                               token_type="Bearer", expires_in=3600))
        self.assertEqual(await asyncio.gather(self.client.access_token(session), self.client.access_token(session)),
                         ["new-access", "new-access"])
        session.post.assert_called_once()
        form = session.post.call_args.kwargs["data"]
        self.assertEqual(form["client_id"], "saved-client")
        self.assertEqual(form["refresh_token"], "old-refresh")
        self.assertNotIn("scope", form)
        again = ChatGPT(self.temp.name)
        self.assertEqual(await again.access_token(session), "new-access")
        self.assertEqual(again.load()["profiles"]["saved-client"]["refresh_token"], "new-refresh")

    async def test_identity_only_grant_never_calls_model_or_refresh(self):
        self.record["scopes"] = ["openid"]
        write_json(self.client.path, {"active": "saved-client", "profiles": {"saved-client": self.record}})
        session = MagicMock()
        with self.assertRaises(ChatGPTError):
            await self.client.answer(session, [{"role": "user", "content": "hello"}], "help")
        session.get.assert_not_called()
        session.post.assert_not_called()

    async def test_response_contract_uses_available_model_and_complete_history(self):
        self.record["expires_at"] = time.time() + 3600
        write_json(self.client.path, {"active": "saved-client", "profiles": {"saved-client": self.record}})
        session = MagicMock()
        session.get.return_value = Reply({"models": [{"slug": "hidden", "visibility": "hidden"},
            {"slug": "available", "display_name": "Available", "visibility": "list"}]})
        response = Reply({})
        response.content = sse([{"type": "response.output_text.delta", "delta": "answer"},
                                {"type": "response.completed", "response": {"usage": {"input_tokens": 5}}}])
        session.post.return_value = response
        history = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "prior"},
                   {"role": "user", "content": "follow up"}]
        result, usage = await self.client.answer(session, history, "instructions")
        self.assertEqual(result, "answer")
        self.assertEqual(usage["input_tokens"], 5)
        call = session.post.call_args
        self.assertEqual(call.args[0], "https://api.openai.com/v1/responses")
        self.assertEqual(call.kwargs["json"], dict(model="available", instructions="instructions", input=history,
                                                 store=False, stream=True))
        self.assertEqual(call.kwargs["timeout"].total, 150)
        self.assertEqual(call.kwargs["timeout"].sock_read, 60)
        response.content = sse([{"type": "response.completed", "response": {"output": [
            {"type": "message", "content": [{"type": "output_text", "text": "answer"}], "role": "assistant"}]}}])
        await self.client.answer(session, history, "instructions", timeout_seconds=240)
        self.assertEqual(session.post.call_args.kwargs["timeout"].total, 240)
        self.assertIsNone(session.post.call_args.kwargs["timeout"].sock_read)

    async def test_stream_failure_and_interruption_never_return_partial_answer(self):
        partial = {"type": "response.output_text.delta", "delta": "partial"}
        for tail in ([], [{"type": "response.incomplete"}], [{"type": "response.failed", "response": {
                "error": {"code": "subscription_sharing_usage_limit_exceeded"}}}]):
            with self.subTest(tail=tail), self.assertRaises(ChatGPTError):
                await collect_response(sse([partial] + tail))

    async def test_reasoning_text_is_not_returned(self):
        text, _ = await collect_response(sse([{"type": "response.reasoning_text.delta", "delta": "private"},
            {"type": "response.completed", "response": {"status": "completed", "output": [
                {"type": "reasoning", "content": [{"type": "output_text", "text": "private"}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "final"}]}]}}]))
        self.assertEqual(text, "final")

    async def test_transfer_moves_login_and_keeps_destination_host_identity(self):
        self.record["expires_at"] = time.time() + 3600
        write_json(self.client.path, {"active": "saved-client", "profiles": {"saved-client": self.record}})
        source_host = self.client.host_id()
        transfer = self.client.export_credentials(move=True)
        exported = json.loads(transfer.read_text(encoding="utf-8"))
        self.assertNotIn(source_host, json.dumps(exported))
        source = self.client.load()["profiles"]["saved-client"]
        self.assertTrue(source["transferred"])
        self.assertNotIn("refresh_token", source)
        session = MagicMock()
        with self.assertRaises(ChatGPTError):
            await self.client.access_token(session)
        with tempfile.TemporaryDirectory() as destination:
            target = ChatGPT(destination)
            destination_host = target.host_id()
            target.import_credentials(transfer)
            self.assertEqual(target.host_id(), destination_host)
            self.assertNotEqual(destination_host, source_host)
            self.assertEqual(await target.access_token(session), "old-access")
            self.assertEqual(target.load()["profiles"]["saved-client"]["refresh_token"], "old-refresh")
        session.post.assert_not_called()

    async def test_invalid_transfer_does_not_replace_saved_login(self):
        before = self.client.path.read_bytes()
        transfer = self.client.directory / "invalid-transfer.json"
        write_json(transfer, {"format": "qqbotgpt-chatgpt-v1", "profile": {**self.record, "issuer": "https://example.com"}})
        with self.assertRaises(ChatGPTError):
            self.client.import_credentials(transfer)
        self.assertEqual(self.client.path.read_bytes(), before)

    async def test_reasoning_is_sent_and_unsupported_effort_is_rejected_before_inference(self):
        self.record["expires_at"] = time.time() + 3600
        write_json(self.client.path, {"active": "saved-client", "profiles": {"saved-client": self.record}})
        session = MagicMock()
        session.get.return_value = Reply({"models": [
            {"slug": "first", "visibility": "list", "supported_reasoning_levels": [{"effort": "low"}, {"effort": "high"}, {"effort": "ultra"}]},
            {"slug": "second", "visibility": "list", "supported_reasoning_levels": [{"effort": "low"}]}]})

        def reply(*args, **kwargs):
            response = Reply({})
            response.content = sse([{"type": "response.completed", "response": {"output": [
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}]}}])
            return response
        session.post.side_effect = reply
        await self.client.answer(session, [], "help", "second", reasoning_effort="low")
        self.assertEqual(session.post.call_args.kwargs["json"]["reasoning"], {"effort": "low"})
        await self.client.answer(session, [], "help")
        self.assertEqual(session.post.call_args.kwargs["json"]["model"], "first")
        self.assertNotIn("reasoning", session.post.call_args.kwargs["json"])
        for effort in ("none", "ultra", "invalid"):
            with self.assertRaises(ChatGPTError):
                await self.client.answer(session, [], "help", reasoning_effort=effort)
        self.assertEqual(session.post.call_count, 2)
