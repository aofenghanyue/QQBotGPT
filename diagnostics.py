"""Bounded, private QQ send-failure records and a read-only inspection command."""
import argparse
import datetime as dt
import json
from pathlib import Path
import re
import sqlite3
import time


RETENTION_SECONDS = 7 * 86400
MAX_RECORDS = 100
MAX_TEXT = 65536
SECRET_KEYS = re.compile(r"authorization|token|secret|password|cookie|api[_-]?key", re.I)


def safe_detail(value, depth=0):
    """Keep error data useful without persisting credential fields or headers."""
    if depth > 8:
        return "[depth limit]"
    if isinstance(value, dict):
        return {str(key): "[redacted]" if SECRET_KEYS.search(str(key)) else safe_detail(item, depth + 1)
                for key, item in list(value.items())[:100]}
    if isinstance(value, list):
        return [safe_detail(item, depth + 1) for item in value[:100]]
    if isinstance(value, str):
        value = re.sub(r"(?i)\b(?:Bearer|QQBot)\s+[A-Za-z0-9._~+/=-]+", "[redacted authorization]", value)
        value = re.sub(r'''(?i)\b(access_token|refresh_token|app_secret|client_secret|api_key|password)["']?\s*[:=]\s*["']?[^\s,"'}<]+''',
                       r"\1=[redacted]", value)
        return value[:8192]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return safe_detail(str(value), depth + 1)


class SendFailureStore:
    def __init__(self, db):
        self.db = db
        db.execute("""CREATE TABLE IF NOT EXISTS send_failures (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL NOT NULL,
            conversation TEXT NOT NULL, message_id TEXT NOT NULL, stage TEXT NOT NULL,
            content TEXT NOT NULL, answer TEXT, details TEXT NOT NULL)""")
        self.prune()

    def prune(self, now=None):
        now = time.time() if now is None else now
        self.db.execute("DELETE FROM send_failures WHERE created < ?", (now - RETENTION_SECONDS,))
        self.db.execute("DELETE FROM send_failures WHERE id NOT IN "
                        "(SELECT id FROM send_failures ORDER BY created DESC,id DESC LIMIT ?)", (MAX_RECORDS,))
        self.db.commit()

    def record(self, conversation, message_id, stage, content, answer, details):
        details = safe_detail(details)
        details["content_truncated"] = len(content) > MAX_TEXT
        details["answer_truncated"] = answer is not None and len(answer) > MAX_TEXT
        encoded = json.dumps(details, ensure_ascii=False)
        if len(encoded) > MAX_TEXT:
            # Preserve the error summary and trace ID if the upstream body is excessive.
            details["response"] = {"truncated": True, "preview": str(details.get("response", ""))[:8192]}
            encoded = json.dumps(details, ensure_ascii=False)
        row = self.db.execute("INSERT INTO send_failures "
                              "(created,conversation,message_id,stage,content,answer,details) VALUES (?,?,?,?,?,?,?)",
                              (time.time(), conversation, message_id, stage, content[:MAX_TEXT],
                               answer[:MAX_TEXT] if answer is not None else None, encoded))
        self.prune()
        return row.lastrowid

    def clear(self, conversation):
        self.db.execute("DELETE FROM send_failures WHERE conversation=?", (conversation,))
        self.db.commit()


def read_failures(path, limit=5):
    if not 1 <= limit <= MAX_RECORDS:
        raise ValueError("limit must be between 1 and 100")
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='send_failures'").fetchone():
            return []
        rows = conn.execute("SELECT * FROM send_failures WHERE created >= ? ORDER BY created DESC,id DESC LIMIT ?",
                            (time.time() - RETENTION_SECONDS, limit)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["created"] = dt.datetime.fromtimestamp(item["created"], dt.timezone.utc).isoformat()
            item["details"] = json.loads(item["details"])
            result.append(item)
        return result
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="查看最近七天的 QQ 发送失败记录（包含私人聊天正文，仅供管理员排查）。")
    parser.add_argument("--data-dir", default=".data", help="机器人数据目录")
    parser.add_argument("--limit", type=int, default=5, help="最多显示 1～100 条，默认 5 条")
    args = parser.parse_args()
    try:
        print(json.dumps(read_failures(Path(args.data_dir) / "state.db", args.limit), ensure_ascii=False, indent=2))
    except (ValueError, sqlite3.Error) as exc:
        parser.exit(1, f"无法读取诊断记录：{exc}\n")


if __name__ == "__main__":
    main()
