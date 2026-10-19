"""ClawBot direct chats: QR login, independent accounts, persistent text replies.

Wire format checked against Tencent/openclaw-weixin 2.4.10; see docs/微信接入.md.
Only the owner returned by QR confirmation can use each account.
"""
import argparse
import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import re
import secrets
import time
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
from filelock import FileLock, Timeout

from bot import ApiError, Bot, QQ_TEXT_STYLE, SYSTEM, load_config, qq_reply, test_model, wants_sources
from chatgpt import ChatGPT, ChatGPTError, secure_directory, write_json

LOG = logging.getLogger("weixin")
API_BASE = "https://ilinkai.weixin.qq.com"
PROTOCOL_VERSION = "2.4.10"
HELP = """直接发送文字即可聊天。

/帮助  查看命令
/状态  查看运行状态
/模型  查看模型及切换用法
/模型 列表  查看可用模型
/思考  查看思考强度
/思考 高  使用高强度思考
/记忆  查看记忆设置
/记忆 200  保留最近200条消息
/记忆 过期 360  闲置6小时清空
/清空  清除当前聊天记录

需要搜索时，直接说“联网查一下……”。
当前仅支持文字私聊。
各微信号的记忆和设置独立。"""


class WeixinError(ApiError):
    """Safe diagnostic; never include response bodies, URLs, tokens or messages."""


class LoginExpired(WeixinError):
    pass


def account_name(value):
    if (not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", value)
            or value in {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
                         *(f"lpt{i}" for i in range(1, 10))}):
        raise ValueError("账号别名请使用小写字母开头的1～32位字母、数字、下划线或短横线，并避开系统保留名。")
    return value


def api_base(value):
    """Do not forward a bearer token to an arbitrary redirect or saved host."""
    if not isinstance(value, str):
        raise WeixinError("微信接口地址无效。")
    try:
        url = urlsplit(value)
        valid = (url.scheme == "https" and url.hostname
                 and (url.hostname == "weixin.qq.com" or url.hostname.endswith(".weixin.qq.com"))
                 and url.port in (None, 443) and not url.username and not url.password
                 and url.path in ("", "/") and not url.query and not url.fragment)
    except ValueError:
        valid = False
    if not valid:
        raise WeixinError("微信接口地址校验失败，需为微信官方 HTTPS 地址。")
    return value.rstrip("/")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def credentials(data):
    if not isinstance(data, dict) or not all(
            isinstance(data.get(key), str) and data[key].strip()
            for key in ("bot_token", "bot_id", "user_id", "base_url")):
        raise WeixinError("微信登录凭据不完整，请重新扫码。")
    return {**data, "base_url": api_base(data["base_url"])}


class WeixinAPI:
    def __init__(self, http, base_url=API_BASE, token=""):
        self.http = http
        self.base_url = api_base(base_url)
        self.token = token

    async def request(self, endpoint, body=None, *, params=None, timeout=20):
        headers = {"iLink-App-Id": "bot", "iLink-App-ClientVersion": str((2 << 16) | (4 << 8) | 10)}
        if body is not None:
            headers.update({"AuthorizationType": "ilink_bot_token",
                            "X-WECHAT-UIN": base64.b64encode(str(secrets.randbits(32)).encode()).decode()})
            if self.token:
                headers["Authorization"] = "Bearer " + self.token
            body = {**body, "base_info": {"channel_version": PROTOCOL_VERSION, "bot_agent": "QQBotGPT/1.0"}}
        async with self.http.request("POST" if body is not None else "GET",
                                     self.base_url + "/ilink/bot/" + endpoint,
                                     params=params, json=body, headers=headers,
                                     timeout=aiohttp.ClientTimeout(total=timeout),
                                     allow_redirects=False) as response:
            if response.status in (401, 403):
                raise LoginExpired("微信授权失效或受限，请重新扫码。")
            if response.status != 200:
                raise WeixinError(f"微信接口 HTTP={response.status}")
            try:
                # iLink serves valid JSON as application/octet-stream in production.
                data = await response.json(content_type=None)
            except (ValueError, aiohttp.ContentTypeError):
                raise WeixinError("微信接口返回了无效数据。") from None
        if not isinstance(data, dict):
            raise WeixinError("微信接口返回格式无效。")
        codes = (data.get("ret", 0), data.get("errcode", 0))
        if any(str(code) == "-14" for code in codes):
            raise LoginExpired("微信登录已过期，请重新扫码。")
        if any(code not in (0, "0", None) for code in codes):
            # Error text can contain private request data; emit only numeric codes.
            safe = ",".join(str(code) for code in codes if re.fullmatch(r"-?\d{1,10}", str(code)))
            raise WeixinError("微信接口请求失败，错误码=" + (safe or "未知"))
        return data

    async def updates(self, cursor, timeout=35):
        try:
            return await self.request("getupdates", {"get_updates_buf": cursor}, timeout=timeout)
        except asyncio.TimeoutError:
            return {"msgs": [], "get_updates_buf": cursor}

    async def send(self, message, reply, client_id):
        if not message.get("context_token"):
            raise WeixinError("消息缺少回复上下文。")
        msg = {"from_user_id": "", "to_user_id": message["from_user_id"], "client_id": client_id,
               "message_type": 2, "message_state": 2, "context_token": message["context_token"],
               "item_list": [{"type": 1, "text_item": {"text": reply}}]}
        if message.get("run_id"):
            msg["run_id"] = message["run_id"]
        return await self.request("sendmessage", {"msg": msg})


def save_account(root, alias, record):
    record = credentials(record)
    for path in root.glob("*/credentials.json"):
        if path.parent.name == alias:
            continue
        other = credentials(read_json(path))
        if (other["bot_id"].casefold() == record["bot_id"].casefold()
                or other["user_id"].casefold() == record["user_id"].casefold()):
            raise WeixinError(f"这个微信号已保存在 {path.parent.name} 下，请用该别名重新登录。")
    folder = secure_directory(root / account_name(alias))
    write_json(folder / "credentials.json", record)
    write_json(folder / "status.json", {"state": "linked", "updated": time.time()})


async def login(root, alias, *, show_qr=None, verify_code=None):
    """Only replace an existing account after complete, validated confirmation."""
    import qrcode
    alias = account_name(alias)
    folder = secure_directory(root / alias)

    def display(value):
        qr = qrcode.QRCode(border=4)
        qr.add_data(value)
        qr.make(fit=True)
        qr.make_image().save(folder / "login.png")
        qr.print_ascii(invert=True)
        print(f"请用 {alias} 对应的微信扫码。二维码图片：{folder / 'login.png'}", flush=True)

    async def enter_code():
        return await asyncio.to_thread(input, "请输入手机微信显示的连接验证码：")

    show_qr = show_qr or display
    verify_code = verify_code or enter_code
    async with aiohttp.ClientSession() as http:
        api = WeixinAPI(http)
        tokens = [credentials(read_json(path))["bot_token"] for path in root.glob("*/credentials.json")
                  if path.parent.name != alias]
        try:
            for _ in range(3):
                api.base_url = API_BASE
                qr = await api.request("get_bot_qrcode", {"local_token_list": tokens[-10:]}, params={"bot_type": "3"})
                if not all(isinstance(qr.get(k), str) and qr[k] for k in ("qrcode", "qrcode_img_content")):
                    raise WeixinError("微信未返回有效二维码。")
                show_qr(qr["qrcode_img_content"])
                deadline, code = time.monotonic() + 180, ""
                while time.monotonic() < deadline:
                    params = {"qrcode": qr["qrcode"]}
                    if code:
                        params["verify_code"] = code
                    try:
                        result = await api.request("get_qrcode_status", params=params, timeout=35)
                    except asyncio.TimeoutError:
                        continue
                    state = result.get("status")
                    if state == "confirmed":
                        record = {"bot_token": result.get("bot_token"), "bot_id": result.get("ilink_bot_id"),
                                  "user_id": result.get("ilink_user_id"),
                                  "base_url": result.get("baseurl") or api.base_url,
                                  "login_id": secrets.token_hex(16), "saved_at": time.time()}
                        save_account(root, alias, record)
                        print(f"{alias} 登录成功；只接受此微信号的私聊。", flush=True)
                        return
                    if state == "need_verifycode":
                        code = (await verify_code()).strip()
                        if not re.fullmatch(r"\d{1,12}", code):
                            raise WeixinError("连接验证码须为手机微信显示的数字。")
                        continue
                    if state == "scaned_but_redirect":
                        api.base_url = api_base("https://" + str(result.get("redirect_host", "")))
                    elif state in ("scaned", "scanned"):
                        code = ""
                    elif state == "expired":
                        print("二维码已过期，正在重新生成。", flush=True)
                        break
                    elif state == "verify_code_blocked":
                        raise WeixinError("验证码多次输入错误，请稍后重新登录。")
                    elif state == "binded_redirect":
                        raise WeixinError("这个微信号已绑定现有连接，请使用原账号别名。")
                    elif state != "wait":
                        raise WeixinError("微信返回了未支持的登录状态，请更新接入程序。")
                    await asyncio.sleep(1)
            raise WeixinError("扫码登录超时，请重新运行 login。")
        finally:
            (folder / "login.png").unlink(missing_ok=True)


def private_message(message, record):
    """Reject group/bot/foreign/incomplete messages before storage or model use."""
    if not isinstance(message, dict):
        return False
    sender = message.get("from_user_id")
    recipient = message.get("to_user_id")
    if (message.get("group_id") or message.get("chat_type") in ("group", "room")
            or message.get("message_type") != 1 or message.get("message_state", 2) != 2
            or message.get("delete_time_ms")
            or not isinstance(sender, str) or sender.casefold() != record["user_id"].casefold()
            or (recipient and (not isinstance(recipient, str) or recipient.casefold() != record["bot_id"].casefold()))
            or not isinstance(message.get("context_token"), str) or not message["context_token"]):
        return False
    mid = message.get("message_id", message.get("msg_id"))
    stamp = message.get("create_time_ms")
    return (isinstance(mid, (str, int)) and not isinstance(mid, bool) and bool(str(mid))
            and isinstance(stamp, (int, float)) and not isinstance(stamp, bool)
            and -60 <= time.time() - stamp / 1000 <= 3600)


class WeixinBot(Bot):
    def __init__(self, config, folder, record, model_client=None, *, state_path=None):
        super().__init__({**config, "owner_only": True, "auto_chat": False, "group_members_can_chat": False},
                         state_path or str(folder / "state.db"))
        self.record = credentials(record)
        self.folder = folder
        self.alias = folder.name
        identity = record["bot_id"].casefold() + "\0" + record["user_id"].casefold()
        self.scope = "private:weixin:" + hashlib.sha256(identity.encode()).hexdigest()
        self.generation = record.get("login_id") or hashlib.sha256(record["bot_token"].encode()).hexdigest()
        if model_client is not None:
            # Share refresh serialization, while keeping selected-model/cache state
            # local to each account so concurrent chats cannot select each other's model.
            self.chatgpt_client().refresh_lock = model_client.refresh_lock
        self.db.execute("CREATE TABLE IF NOT EXISTS wx_inbox (generation TEXT, id TEXT, payload TEXT, reply TEXT, "
                        "chat INTEGER DEFAULT 0, done INTEGER DEFAULT 0, at REAL, PRIMARY KEY(generation,id))")
        self.db.execute("CREATE TABLE IF NOT EXISTS wx_cursor (generation TEXT PRIMARY KEY, cursor TEXT)")
        self.db.execute("DELETE FROM wx_inbox WHERE generation != ?", (self.generation,))
        self.db.execute("DELETE FROM wx_cursor WHERE generation != ?", (self.generation,))
        self.db.commit()

    def answer_prompt(self):
        return ((self.c.get("system_prompt") or SYSTEM.replace("QQ", "微信"))
                + QQ_TEXT_STYLE.replace("QQ", "微信") + "\n当前是微信私聊，只处理文字，不参与群聊。")

    async def command(self, text):
        scope, name = self.scope, text.split()[0]
        if text in ("/帮助", "/help", "帮助"):
            return HELP
        if text in ("/状态", "/status"):
            return (f"微信账号：{self.alias}\n连接：正常；仅本人私聊\n" + self.model_status(scope)
                    + "\n" + self.thinking_status(scope) + "\n" + self.memory_status(scope)
                    + f"\n最近错误：{self.last_error}")
        if name == "/模型":
            return await self.model_command(scope, text, "owner")
        if name == "/思考":
            if self.model_settings(scope)["provider"] == "chatgpt":
                return await self.chatgpt_thinking_command(scope, text, "owner")
            return self.thinking_command(scope, text, "owner")
        if name in ("/记忆", "/上下文"):
            return self.memory_command(scope, text, "owner")
        if text == "/清空":
            self.clear_memory(scope)
            return "当前微信账号的聊天记录已清空，模型和记忆设置保留。"
        return "命令格式有误，请发送 /帮助 查看用法。"

    def ingest(self, batch):
        messages = batch.get("msgs", [])
        cursor = batch.get("get_updates_buf")
        if not isinstance(messages, list) or (cursor is not None and not isinstance(cursor, str)):
            raise WeixinError("微信消息批次格式无效。")
        with self.db:
            for msg in messages:
                if private_message(msg, self.record):
                    mid = str(msg.get("message_id", msg.get("msg_id")))
                    self.db.execute("INSERT OR IGNORE INTO wx_inbox (generation,id,payload,at) VALUES (?,?,?,?)",
                                    (self.generation, mid, json.dumps(msg, ensure_ascii=False), time.time()))
            if cursor is not None:
                self.db.execute("INSERT OR REPLACE INTO wx_cursor VALUES (?,?)", (self.generation, cursor))
            self.db.execute("DELETE FROM wx_inbox WHERE done=1 AND at < ?", (time.time() - 86400,))

    async def process_pending(self, api):
        rows = self.db.execute("SELECT id,payload,reply,chat FROM wx_inbox WHERE generation=? AND done=0 ORDER BY rowid",
                               (self.generation,)).fetchall()
        for mid, payload, reply, chat in rows:
            msg = json.loads(payload)
            if not private_message(msg, self.record):
                self.complete(mid)
                continue
            self.expire_memory(self.scope)
            if reply is None:
                items = msg.get("item_list") or []
                if not isinstance(items, list):
                    items = []
                # Mixed media must not imply that the model has seen an attachment.
                text_only = bool(items) and all(isinstance(item, dict) and item.get("type") == 1
                    and isinstance(item.get("text_item"), dict)
                    and isinstance(item["text_item"].get("text"), str) for item in items)
                text = "\n".join(item["text_item"]["text"] for item in items).strip() if text_only else ""
                sources = []
                if not text_only or not text:
                    reply = "当前微信助手支持文字聊天。请把问题以文字发来，图片、语音和文件暂未接入。"
                elif len(text) > 12000:
                    reply = "这条消息较长，请拆成每条12000字以内再发送。"
                elif text.startswith("/") or text == "帮助":
                    try:
                        reply = await self.command(text)
                    except (ChatGPTError, ApiError) as exc:
                        reply = str(exc)
                    except (aiohttp.ClientError, asyncio.TimeoutError, Timeout):
                        reply = "连接模型服务失败，设置未更改，请稍后重试。"
                    reply = reply.replace("本群", "本会话").replace("群友", "用户").replace("群里", "对话中")
                else:
                    chat = 1
                    memory_id = self.generation + ":" + mid
                    if not self.db.execute("SELECT 1 FROM memory_seen WHERE id=?", (memory_id,)).fetchone():
                        self.append_memory(self.scope, {"role": "user", "content": text})
                        self.db.execute("INSERT INTO memory_seen VALUES (?,?)", (memory_id, time.time()))
                        self.db.execute("DELETE FROM memory_seen WHERE at < ?", (time.time() - 86400,))
                        self.db.commit()
                    try:
                        reply = await self.answer(list(self.history[self.scope]), True, self.scope)
                        sources = self.last_model.get(self.scope, {}).get("web_sources", [])
                    except Exception as exc:
                        self.last_error = (str(exc) if isinstance(exc, ChatGPTError)
                                           else "模型连接失败，请稍后重试。")
                        LOG.warning("weixin_model_failed account=%s type=%s", self.alias, type(exc).__name__)
                        reply = self.last_error
                        chat = 0
                reply = qq_reply(reply, sources, show_sources=self.c.get("chatgpt_show_sources", False)
                                 or wants_sources(text)) or "刚才未能生成有效回复，请再试一次。"
                self.db.execute("UPDATE wx_inbox SET reply=?,chat=? WHERE generation=? AND id=?",
                                (reply, chat, self.generation, mid))
                self.db.commit()
            client_id = "qqbotgpt-" + hashlib.sha256((self.generation + ":" + mid).encode()).hexdigest()[:32]
            await api.send(msg, reply, client_id)
            if chat:
                self.append_memory(self.scope, {"role": "assistant", "content": reply})
            self.complete(mid)
            LOG.info("weixin_reply_sent account=%s", self.alias)

    def complete(self, mid):
        self.db.execute("UPDATE wx_inbox SET done=1,payload='',reply=NULL WHERE generation=? AND id=?",
                        (self.generation, mid))
        self.db.commit()

    def set_status(self, state):
        path = self.folder / "status.json"
        previous = read_json(path, {})
        if previous.get("state") != state:
            write_json(path, {"state": state, "updated": time.time()})

    async def run_private(self):
        delay, timeout = 2, 35
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as self.http:
            api = WeixinAPI(self.http, self.record["base_url"], self.record["bot_token"])
            try:
                while True:
                    try:
                        await self.process_pending(api)
                        self.expire_memory(self.scope)
                        row = self.db.execute("SELECT cursor FROM wx_cursor WHERE generation=?", (self.generation,)).fetchone()
                        batch = await api.updates(row[0] if row else "", timeout)
                        self.ingest(batch)
                        suggested = batch.get("longpolling_timeout_ms")
                        if isinstance(suggested, (int, float)) and 1000 <= suggested <= 60000:
                            timeout = suggested / 1000 + 5
                        self.connected = True
                        self.set_status("connected")
                        delay = 2
                        await asyncio.sleep(0.2)
                    except LoginExpired:
                        self.set_status("login_required")
                        LOG.error("weixin_login_required account=%s；请用此别名重新运行 login", self.alias)
                        return
                    except (WeixinError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
                        self.connected = False
                        self.set_status("retrying")
                        LOG.warning("weixin_retry account=%s type=%s delay=%s", self.alias, type(exc).__name__, delay)
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 60)
            finally:
                self.connected = False


async def serve(config, root, only=None):
    """One task per account; new or refreshed credentials are loaded automatically."""
    tasks, revisions = {}, {}
    model_client = ChatGPT(config["data_dir"])

    async def run_account(path, record):
        bot = WeixinBot(config, path.parent, record, model_client)
        try:
            LOG.info("weixin_account_started account=%s", path.parent.name)
            await bot.run_private()
        finally:
            bot.db.close()

    try:
        while True:
            paths = {path.parent.name: path for path in root.glob("*/credentials.json")
                     if only is None or path.parent.name == only}
            for alias in list(tasks):
                if alias not in paths:
                    tasks[alias].cancel()
                    await asyncio.gather(tasks.pop(alias), return_exceptions=True)
                    revisions.pop(alias, None)
            for alias, path in paths.items():
                try:
                    account_name(alias)
                    raw = path.read_bytes()
                    revision = hashlib.sha256(raw).hexdigest()
                    if revisions.get(alias) == revision:
                        continue
                    record = credentials(json.loads(raw.decode("utf-8")))
                except (ValueError, WeixinError):
                    LOG.error("weixin_invalid_account account=%s", alias)
                    continue
                if alias in tasks:
                    tasks[alias].cancel()
                    await asyncio.gather(tasks[alias], return_exceptions=True)
                revisions[alias] = revision
                tasks[alias] = asyncio.create_task(run_account(path, record))
            for alias, task in tasks.items():
                if task.done() and not task.cancelled() and task.exception():
                    # A broken database or unexpected programming error must be visible.
                    raise WeixinError(f"微信账号 {alias} 运行中止（{type(task.exception()).__name__}）。")
            await asyncio.sleep(3)
    finally:
        for task in tasks.values():
            task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description="微信 ClawBot 文字私聊助手，支持多个微信号")
    parser.add_argument("--config", default="config.json", help="模型配置文件；无需 QQ 凭据")
    sub = parser.add_subparsers(dest="command", required=True)
    auth = sub.add_parser("login", help="用指定微信号扫码连接")
    auth.add_argument("--account", required=True, type=account_name, help="本地别名，例如 main 或 second")
    sub.add_parser("accounts", help="列出已连接的本地账号别名，不显示凭据")
    run = sub.add_parser("run", help="运行全部账号，自动加载新增账号")
    run.add_argument("--account", type=account_name, help="只运行指定账号")
    sub.add_parser("test-model", help="测试模型连接，不连接微信")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config(args.config, require_qq=False)
        root = secure_directory(Path(config["data_dir"]) / "weixin")
        if args.command == "login":
            with FileLock(str(root / "login.lock"), timeout=0):
                asyncio.run(login(root, args.account))
        elif args.command == "accounts":
            paths = sorted(root.glob("*/credentials.json"))
            for path in paths:
                credentials(read_json(path))
                state = read_json(path.parent / "status.json", {}).get("state", "linked")
                print(f"{path.parent.name}：已保存登录；最近状态={state}（不代表当前在线）")
            if not paths:
                print("尚无微信账号，请先运行 login --account main。")
        elif args.command == "test-model":
            bot = WeixinBot(config, root, {"bot_id": "test", "user_id": "test", "bot_token": "test", "base_url": API_BASE},
                            state_path=":memory:")
            try:
                asyncio.run(test_model(bot))
            finally:
                bot.db.close()
        else:
            if args.account and not (root / args.account / "credentials.json").exists():
                raise WeixinError("此账号尚未登录，请先运行 login。")
            with FileLock(str(root / "run.lock"), timeout=0):
                print("微信私聊服务已启动；新增账号扫码后会自动加载。按 Ctrl+C 停止。", flush=True)
                asyncio.run(serve(config, root, args.account))
    except Timeout:
        parser.exit(1, "已有微信服务或扫码登录进程运行，请使用现有进程。\n")
    except FileNotFoundError:
        parser.exit(1, "未找到配置或模型凭据文件，请按 docs/微信接入.md 配置。\n")
    except (ValueError, ChatGPTError, WeixinError) as exc:
        parser.exit(1, f"{exc}\n")
    except (aiohttp.ClientError, asyncio.TimeoutError):
        parser.exit(1, "微信或模型服务连接失败，请检查网络后重试。\n")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
