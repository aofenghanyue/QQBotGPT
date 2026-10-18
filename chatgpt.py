"""Official Sign in with ChatGPT and text Responses API, with a small CLI."""
import argparse
import asyncio
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import tempfile
import time
from urllib.parse import urlencode, urlsplit
import uuid
import webbrowser

import aiohttp
from aiohttp import web
from filelock import FileLock, Timeout
import jwt

ISSUER = "https://auth.openai.com"
AUTHORIZE = ISSUER + "/api/accounts/authorize"
TOKEN = ISSUER + "/api/accounts/oauth/token"
RESOURCE = "https://api.openai.com/v1"
DIRECT_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke " + DIRECT_SCOPE
MANAGE_USAGE = "https://chatgpt.com/#settings/Usage"
API_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def supported_efforts(model):
    # The account catalog can also contain app-only delegation modes.
    levels = model.get("supported_reasoning_levels", [])
    return [level.get("effort") if isinstance(level, dict) else level for level in levels
            if (level.get("effort") if isinstance(level, dict) else level) in API_EFFORTS]


class ChatGPTError(Exception):
    """A safe, user-facing error, without tokens or arbitrary response bodies."""


def secure_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "nt":
        identity = subprocess.check_output(
            ["whoami", "/user", "/fo", "csv", "/nh"], text=True).strip()
        sid = next(csv.reader([identity]))[1]
        subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r",
                        f"*{sid}:(OI)(CI)F", "*S-1-5-18:(OI)(CI)F"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    else:
        path.chmod(0o700)
    return path


def write_json(path, data):
    """Atomic replacement; the containing directory is already protected."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if os.name != "nt":
            os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path, default):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default


def error_code(data):
    error = data.get("error", data) if isinstance(data, dict) else {}
    code = error.get("code", error.get("type", "request_failed")) if isinstance(error, dict) else error
    return str(code) if re.fullmatch(r"[a-zA-Z0-9_.-]{1,100}", str(code)) else "request_failed"


def api_error(data, status=0):
    code = error_code(data)
    if "usage_limit" in code or "usage_unavailable" in code or status == 429:
        return ChatGPTError(f"ChatGPT 额度暂不可用（{code}）。请在 ChatGPT 设置 → Usage 查看额度。")
    if code in ("invalid_grant", "invalid_token") or status == 401:
        return ChatGPTError("ChatGPT 登录已失效，请运行 python chatgpt.py login 重新登录。")
    if "not_eligible" in code or "unsupported_country" in code or status == 403:
        return ChatGPTError(f"ChatGPT 未允许这次调用（{code}，HTTP {status}），请检查账户权限和服务可用地区。")
    return ChatGPTError(f"ChatGPT 请求失败（{code}，HTTP {status}）。")


async def response_json(response):
    try:
        data = await response.json(content_type=None)
    except (ValueError, UnicodeError):
        raise ChatGPTError(f"服务返回了无法读取的响应（HTTP {response.status}）。") from None
    if response.status >= 300:
        raise api_error(data, response.status)
    return data


def validate_callback(query, state, client_id):
    if not secrets.compare_digest(str(query.get("state", "")), state):
        raise ChatGPTError("登录回调校验失败，请重新登录。")
    if query.get("error"):
        raise ChatGPTError("登录未完成或授权被取消，请重新运行登录命令。")
    returned = query.get("client_id")
    if client_id != "dynamic_agent_client":
        if returned and returned != client_id:
            raise ChatGPTError("回调中的账户注册与本次登录不匹配。")
        returned = client_id
    if not returned or returned == "dynamic_agent_client" or not query.get("code"):
        raise ChatGPTError("登录回调缺少授权码或已注册的 client_id。")
    return returned, query["code"]


def validate_identity(id_token, jwks, client_id, nonce, previous=None):
    try:
        header = jwt.get_unverified_header(id_token)
        key = next(k for k in jwks["keys"] if k.get("kid") == header.get("kid"))
        claims = jwt.decode(id_token, jwt.PyJWK.from_dict(key).key,
                            algorithms=["RS256"], audience=client_id, issuer=ISSUER,
                            options={"require": ["exp", "iat", "iss", "aud", "sub", "nonce"]})
        if not secrets.compare_digest(str(claims["nonce"]), nonce):
            raise ValueError("nonce")
        if previous and (claims["sub"] != previous["subject"] or claims["iss"] != previous["issuer"]):
            raise ValueError("identity")
        return claims
    except (jwt.PyJWTError, StopIteration, KeyError, ValueError, TypeError):
        raise ChatGPTError("ChatGPT 身份校验失败，未替换已保存的登录。") from None


def credential_record(data, client_id, claims, previous=None):
    if not data.get("access_token") or str(data.get("token_type", "")).lower() != "bearer":
        raise ChatGPTError("授权响应缺少有效访问凭据。")
    record = dict(previous or {})
    record.update(client_id=client_id, issuer=claims["iss"], subject=claims["sub"],
                  email=claims.get("email", ""), access_token=data["access_token"],
                  token_type="Bearer", expires_at=time.time() + int(data["expires_in"]),
                  scopes=data.get("scope", "").split(), earliest_refresh_at=data.get("earliest_refresh_at"))
    for name in ("id_token", "refresh_token"):
        if data.get(name):
            record[name] = data[name]
    return record


class ChatGPT:
    def __init__(self, data_dir=".data"):
        self.directory = secure_directory(data_dir)
        self.path = self.directory / "chatgpt.json"
        self.lock = FileLock(str(self.directory / "chatgpt.lock"), timeout=0)
        self.refresh_lock = asyncio.Lock()
        self.selected_model = ""
        self.catalog = []
        self.catalog_until = 0
        self.catalog_account = None

    def load(self):
        return read_json(self.path, {"profiles": {}, "active": None, "pending": None})

    def host_id(self):
        path = self.directory / "host.json"
        record = read_json(path, {})
        if not record.get("ext_agent_host_id"):
            record = {"ext_agent_host_id": "urn:uuid:" + str(uuid.uuid4())}
            write_json(path, record)
        return record["ext_agent_host_id"]

    def export_credentials(self, move=False):
        with self.lock:
            saved = self.load()
            record = saved["profiles"].get(saved.get("active"))
            if not record or not record.get("refresh_token") or record.get("transferred"):
                raise ChatGPTError("没有可转移的登录，请先完成登录。")
            path = self.directory / "chatgpt-transfer.json"
            write_json(path, {"format": "qqbotgpt-chatgpt-v1", "profile": record})
            if move:
                for key in ("access_token", "refresh_token", "id_token"):
                    record.pop(key, None)
                record["transferred"] = True
                write_json(self.path, saved)
            return path

    def import_credentials(self, path):
        incoming = read_json(path, {})
        record = incoming.get("profile", {})
        required = ("client_id", "issuer", "subject", "access_token", "refresh_token", "expires_at", "scopes")
        if (incoming.get("format") != "qqbotgpt-chatgpt-v1" or any(key not in record for key in required)
                or record["issuer"] != ISSUER or record["client_id"] == "dynamic_agent_client"
                or not record["access_token"] or not record["refresh_token"]):
            raise ChatGPTError("登录转移文件无效，请用 export 命令重新导出。")
        with self.lock:
            # The destination host keeps its own identity, never the source host's.
            self.host_id()
            saved = self.load()
            record.pop("transferred", None)
            saved["profiles"][record["client_id"]] = record
            saved.update(active=record["client_id"], pending=None)
            write_json(self.path, saved)

    async def login(self, session, new=False, open_browser=True, timeout=600):
        # One process owns registration or token rotation at a time.
        with self.lock:
            saved = self.load()
            selected = None if new else saved.get("active") or saved.get("pending")
            previous = saved["profiles"].get(selected)
            if previous and previous.get("transferred"):
                raise ChatGPTError("这份登录已转交服务器。本机另建连接请运行 python chatgpt.py login --new。")
            client_id = selected or "dynamic_agent_client"
            state, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(3))
            challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
            result = asyncio.get_running_loop().create_future()

            async def callback(request):
                # A stray request must not consume the pending authorization.
                if not secrets.compare_digest(request.query.get("state", ""), state):
                    return web.Response(status=400, text="Invalid state. Return to the login page.")
                if not result.done():
                    result.set_result(dict(request.query))
                return web.Response(text="Login received. Return to QQBotGPT to check the result.")

            app = web.Application()
            url = ""
            start_path = "/login/" + secrets.token_urlsafe(24)
            async def start(_request):
                raise web.HTTPFound(url)
            app.router.add_get("/auth/callback", callback)
            app.router.add_get(start_path, start)
            runner = web.AppRunner(app, access_log=None)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = runner.addresses[0][1]
            redirect = f"http://127.0.0.1:{port}/auth/callback"
            params = dict(client_id=client_id, ext_agent_host_id=self.host_id(), response_type="code",
                          redirect_uri=redirect, scope=SCOPES, resource=RESOURCE, state=state,
                          nonce=nonce, code_challenge_method="S256", code_challenge=challenge)
            if client_id == "dynamic_agent_client":
                params["agent_name_hint"] = "QQBotGPT"
            elif previous:
                if previous.get("id_token"):
                    params["id_token_hint"] = previous["id_token"]
                elif previous.get("email"):
                    params["login_hint"] = previous["email"]
            url = AUTHORIZE + "?" + urlencode(params)
            # This loopback URL keeps identity hints out of console logs.
            start_url = f"http://127.0.0.1:{port}" + start_path
            try:
                print("Continue with ChatGPT：请在这台电脑的浏览器中完成登录：\n" + start_url, flush=True)
                if open_browser:
                    await asyncio.to_thread(webbrowser.open, start_url)
                try:
                    query = await asyncio.wait_for(result, timeout)
                except asyncio.TimeoutError:
                    raise ChatGPTError("等待登录超时，请重新运行登录命令。") from None
                issued_id, code = validate_callback(query, state, client_id)
                saved["pending"] = issued_id
                write_json(self.path, saved)
                async with session.post(TOKEN, data=dict(grant_type="authorization_code", client_id=issued_id,
                    code=code, code_verifier=verifier, redirect_uri=redirect, resource=RESOURCE)) as response:
                    data = await response_json(response)
                async with session.get(ISSUER + "/.well-known/jwks.json") as response:
                    jwks = await response_json(response)
                claims = validate_identity(data.get("id_token", ""), jwks, issued_id, nonce, previous)
                saved["profiles"][issued_id] = credential_record(data, issued_id, claims)
                saved.update(active=issued_id, pending=None)
                write_json(self.path, saved)
                if DIRECT_SCOPE not in saved["profiles"][issued_id]["scopes"]:
                    raise ChatGPTError("已登录，但未获准使用 ChatGPT 计划额度；暂时无法调用模型。")
                print("ChatGPT 登录成功，已允许使用计划额度。", flush=True)
            finally:
                await runner.cleanup()

    async def access_token(self, session):
        async with self.refresh_lock:
            return await self._access_token_locked(session)

    async def _access_token_locked(self, session):
        with self.lock:
            saved = self.load()
            record = saved["profiles"].get(saved.get("active"))
            if record and record.get("transferred"):
                raise ChatGPTError("这份登录已转交服务器。本机另建连接请运行 python chatgpt.py login --new。")
            if not record or DIRECT_SCOPE not in record.get("scopes", []):
                raise ChatGPTError("请先运行 python chatgpt.py login，并授权使用 ChatGPT 计划额度。")
            now = time.time()
            if record.get("expires_at", 0) > now + 60:
                return record["access_token"]
            earliest = record.get("earliest_refresh_at")
            if isinstance(earliest, (float, int)) and now < earliest:
                if record.get("expires_at", 0) > now:
                    return record["access_token"]
                raise ChatGPTError("尚未到允许续期的时间，请稍后重试。")
            if not record.get("refresh_token"):
                raise ChatGPTError("授权无法自动续期，请运行 python chatgpt.py login。")
            async with session.post(TOKEN, data=dict(grant_type="refresh_token", client_id=record["client_id"],
                    refresh_token=record["refresh_token"], resource=RESOURCE)) as response:
                data = await response_json(response)
            if not data.get("refresh_token"):
                raise ChatGPTError("续期响应缺少新的续期凭据，请重新登录。")
            # OAuth refresh may omit scope to retain the existing grant.
            data.setdefault("scope", " ".join(record["scopes"]))
            updated = credential_record(data, record["client_id"],
                {"iss": record["issuer"], "sub": record["subject"], "email": record.get("email", "")}, record)
            saved["profiles"][record["client_id"]] = updated
            write_json(self.path, saved)
            if DIRECT_SCOPE not in updated["scopes"]:
                raise ChatGPTError("续期后已失去 ChatGPT 计划调用权限，请重新授权。")
            return updated["access_token"]

    async def models(self, session, refresh=False):
        account = self.load().get("active")
        if not refresh and self.catalog and self.catalog_account == account and time.monotonic() < self.catalog_until:
            return self.catalog
        token = await self.access_token(session)
        async with session.get(RESOURCE + "/models", headers={"Authorization": "Bearer " + token}) as response:
            data = await response_json(response)
        models = [m for m in data.get("models", []) if m.get("visibility") == "list" and m.get("slug")]
        if not models:
            raise ChatGPTError("当前账户没有可用模型。")
        self.catalog, self.catalog_account = models, account
        self.catalog_until = time.monotonic() + 300
        return models

    async def resolve_model(self, session, model=""):
        models = await self.models(session)
        chosen = next((item for item in models if item["slug"] == model), None) if model else models[0]
        if chosen is None:
            raise ChatGPTError("配置的模型不在账户可用列表中，请发送 /模型 列表 查看。")
        return chosen

    async def answer(self, session, history, instructions, model="", reasoning_effort=None, web_search=False):
        chosen = await self.resolve_model(session, model)
        if reasoning_effort is not None and reasoning_effort not in supported_efforts(chosen):
            raise ChatGPTError("当前模型不支持这个思考强度，请发送 /思考 查看可用档位，或发送 /思考 默认。")
        self.selected_model = chosen["slug"]
        token = await self.access_token(session)
        body = dict(model=self.selected_model, instructions=instructions, input=history, store=False, stream=True)
        if reasoning_effort is not None:
            body["reasoning"] = {"effort": reasoning_effort}
        if web_search:
            body["tools"] = [{"type": "web_search"}]
            body["tool_choice"] = "auto"
        async with session.post(RESOURCE + "/responses", json=body,
                headers={"Authorization": "Bearer " + token},
                timeout=aiohttp.ClientTimeout(total=150, sock_read=60)) as response:
            if response.status >= 300:
                await response_json(response)
            return await collect_response(response.content)


def answer_with_citations(output, fallback=""):
    """Keep the final answer and turn verified citation annotations into references."""
    sources, pieces = [], []
    messages = [item for item in output if item.get("type") == "message" and item.get("role") == "assistant"]
    if any(item.get("phase") == "final_answer" for item in messages):
        messages = [item for item in messages if item.get("phase") == "final_answer"]
    for item in messages:
        for part in item.get("content", []):
            if part.get("type") != "output_text":
                continue
            text = part.get("text", "")
            edits = {}
            for annotation in part.get("annotations", []):
                if annotation.get("type") != "url_citation":
                    continue
                url = annotation.get("url", "")
                try:
                    valid = isinstance(url, str) and urlsplit(url).scheme in ("http", "https") and urlsplit(url).hostname
                except ValueError:
                    valid = False
                if not valid or re.search(r"[\s\x00-\x1f]", url):
                    continue
                index = next((i for i, source in enumerate(sources, 1) if source["url"] == url), None)
                if index is None:
                    title = " ".join(str(annotation.get("title") or "网页来源").split())[:80]
                    sources.append({"title": title, "url": url})
                    index = len(sources)
                start, end = annotation.get("start_index"), annotation.get("end_index")
                if not (type(start) is int and type(end) is int and 0 <= start <= end <= len(text)):
                    start = end = len(text)
                span = text[start:end]
                # Replace citation markup only. Some offsets cover a claim; keep that text.
                if not (re.fullmatch(r"\s*\(?\[[^\]]*\]\(https?://[^\n]+\)\)?\s*", span)
                        or re.fullmatch(r"\s*cite[^]+\s*", span)):
                    start = end
                edits.setdefault((start, end), []).append(f"〔{index}〕")
            for (start, end), labels in sorted(edits.items(), reverse=True):
                text = text[:start] + "".join(dict.fromkeys(labels)) + text[end:]
            pieces.append(re.sub(r"cite[^]+", "", text))
    return "\n".join(pieces).strip() or fallback.strip(), sources


async def collect_response(stream):
    """SSE frames can span multiple data lines; partial text is never returned."""
    lines, text, items = [], [], {}
    async for raw_line in stream:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            lines.append(line[5:].lstrip(" "))
        elif not line and lines:
            payload, lines = "\n".join(lines), []
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except ValueError:
                raise ChatGPTError("模型返回了无法读取的流式事件。") from None
            kind = event.get("type")
            if kind == "response.output_text.delta":
                text.append(event.get("delta", ""))
            elif kind == "response.output_item.done":
                item = event.get("item", {})
                if item.get("type") in ("message", "web_search_call"):
                    items[item.get("id", event.get("output_index", len(items)))] = item
            elif kind in ("response.failed", "error"):
                raise api_error(event.get("response", event))
            elif kind == "response.incomplete":
                raise ChatGPTError("这次回答未完成，请缩小问题范围后重试。")
            elif kind == "response.completed":
                result = event.get("response", {})
                if result.get("status", "completed") != "completed":
                    raise ChatGPTError("这次回答未完成。")
                output = result.get("output") or list(items.values())
                final, sources = answer_with_citations(output, "".join(text))
                if not final.strip():
                    raise ChatGPTError("模型完成请求，但没有返回文字回答。")
                usage = dict(result.get("usage") or {})
                usage["web_search_calls"] = sum(item.get("type") == "web_search_call" and item.get("status") == "completed" for item in output)
                usage["web_sources"] = sources
                return final.strip(), usage
    raise ChatGPTError("连接在回答完成前中断，请重试。")


async def cli(args):
    client = ChatGPT(args.data_dir)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        if args.command == "login":
            await client.login(session, new=args.new, open_browser=not args.no_browser)
        elif args.command == "models":
            for model in await client.models(session):
                print(model["slug"], "—", model.get("display_name", model["slug"]))
        elif args.command == "test":
            reply, usage = await client.answer(session, [{"role": "user", "content": args.prompt}],
                                              "请直接、简明地回答。", args.model, web_search=args.search)
            print("使用 ChatGPT 计划；模型：" + client.selected_model)
            print(reply)
            if args.search:
                print("搜索/读取网页次数：" + str(usage["web_search_calls"]))
                for i, source in enumerate(usage["web_sources"], 1):
                    print(f"〔{i}〕 {source['title']}\n{source['url']}")
        elif args.command == "status":
            saved = client.load()
            for key, record in saved["profiles"].items():
                print("*" if key == saved["active"] else " ", key, record.get("email", ""),
                      "已转交服务器" if record.get("transferred") else ("已授权计划额度" if DIRECT_SCOPE in record.get("scopes", []) else "仅登录"))
            if not saved["profiles"]:
                print("尚未登录，请运行 python chatgpt.py login。")
            print("额度管理：" + MANAGE_USAGE)
        elif args.command == "use":
            with client.lock:
                saved = client.load()
                if args.client_id not in saved["profiles"]:
                    raise ChatGPTError("没有找到该登录，运行 python chatgpt.py status 查看。")
                saved["active"] = args.client_id
                write_json(client.path, saved)
            print("已切换登录，请重新启动机器人。")
        elif args.command == "init-host":
            with client.lock:
                client.host_id()
            print("已保存本机身份。")
        elif args.command == "export":
            print("登录转移文件：" + str(client.export_credentials(move=args.move).resolve()))
            if args.move:
                print("本机已停止使用这份登录；请将转移文件安全传至服务器并导入。")
        elif args.command == "import":
            client.import_credentials(args.path)
            print("已导入登录，保留本机身份。请测试连接，成功后删除转移文件。")


def main():
    parser = argparse.ArgumentParser(description="QQBotGPT 的 ChatGPT 登录和连接测试")
    parser.add_argument("--data-dir", default=".data")
    commands = parser.add_subparsers(dest="command", required=True)
    login = commands.add_parser("login", help="Continue with ChatGPT")
    login.add_argument("--new", action="store_true", help="添加另一个账户或工作区")
    login.add_argument("--no-browser", action="store_true")
    commands.add_parser("status")
    commands.add_parser("models")
    commands.add_parser("init-host", help="为当前设备准备独立身份")
    export = commands.add_parser("export", help="导出当前登录供服务器导入")
    export.add_argument("--move", action="store_true", help="导出后停止本机使用这份登录")
    import_command = commands.add_parser("import", help="导入登录并保留当前设备身份")
    import_command.add_argument("path")
    use = commands.add_parser("use")
    use.add_argument("client_id")
    test = commands.add_parser("test")
    test.add_argument("prompt", nargs="?", default="请只回复：连接成功")
    test.add_argument("--model", default="")
    test.add_argument("--search", action="store_true", help="测试官方网页搜索并显示来源")
    args = parser.parse_args()
    try:
        asyncio.run(cli(args))
    except (ChatGPTError, Timeout) as exc:
        parser.exit(1, f"{exc if isinstance(exc, ChatGPTError) else '另一进程正在登录或续期，请稍后重试。'}\n")
    except (aiohttp.ClientError, asyncio.TimeoutError):
        parser.exit(1, "连接服务失败，请检查网络后重试。\n")
    except KeyboardInterrupt:
        parser.exit(130, "已取消。\n")


if __name__ == "__main__":
    main()
