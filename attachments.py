"""Bounded, per-turn attachments. History stores references, never model input blobs."""
import asyncio
import base64
import binascii
import json
import re
import time
from pathlib import PurePosixPath
from urllib.parse import quote, urljoin, urlsplit

import aiohttp
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from chatgpt import ChatGPTError


MAX_BYTES = 20 * 1024 * 1024
PENDING_SECONDS = 300
REFERENCE_SECONDS = 86400
FILE_TYPES = {
    '.pdf': 'application/pdf', '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.xls': 'application/vnd.ms-excel',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.ppt': 'application/vnd.ms-powerpoint',
    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    '.odt': 'application/vnd.oasis.opendocument.text', '.rtf': 'application/rtf',
    '.csv': 'text/csv', '.tsv': 'text/tsv', '.json': 'application/json',
}
for _suffix in ('txt text md markdown log py js ts html htm xml css c h cpp java '
                'go rs sh sql yaml yml toml ini conf rst tex').split():
    FILE_TYPES['.' + _suffix] = 'text/plain'


class AttachmentError(ChatGPTError):
    pass


def bounded(items):
    """Latest kind wins: two images OR one file, never an unbounded history."""
    if not items:
        return []
    kind = items[-1]['kind']
    unique = {}
    for item in items:
        if item['kind'] == kind:
            key = (str(item.get('url')), str(item.get('name')), str(item.get('key')))
            unique.pop(key, None)
            unique[key] = item
    return list(unique.values())[-(2 if kind == 'image' else 1):]


def safe_url(value, platform):
    if not isinstance(value, str):
        raise AttachmentError('附件下载地址无效，请重新发送。')
    if value.startswith('//'):
        value = 'https:' + value
    elif value.startswith('http://'):
        value = 'https://' + value[7:]
    try:
        parsed = urlsplit(value)
        domains = ('weixin.qq.com',) if platform == 'weixin' else ('qq.com', 'qq.com.cn', 'qpic.cn', 'gtimg.cn')
        valid = (parsed.scheme == 'https' and parsed.hostname
                 and any(parsed.hostname == d or parsed.hostname.endswith('.' + d) for d in domains)
                 and parsed.port in (None, 443) and not parsed.username and not parsed.password
                 and not re.search(r'[\s\x00-\x1f\\]', value))
    except ValueError:
        valid = False
    if not valid:
        raise AttachmentError('附件下载地址未通过平台域名校验，请重新发送原附件。')
    return value


def filename(value):
    name = PurePosixPath(str(value or 'file').replace('\\', '/')).name
    return re.sub(r'[\x00-\x1f\x7f]', '', name)[:200] or 'file'


def validate(item):
    result = dict(item)
    result['url'] = safe_url(result.get('url'), result['platform'])
    result['name'] = filename(result.get('name'))
    if result.get('key') and (not isinstance(result['key'], str) or len(result['key']) > 128):
        raise AttachmentError('附件解密信息无效，请重新发送。')
    try:
        size = int(result.get('size') or 0)
    except (TypeError, ValueError):
        raise AttachmentError('附件大小信息无效，请重新发送。') from None
    if size < 0 or size > MAX_BYTES:
        raise AttachmentError('单个图片或文件最多20 MiB，请压缩或拆分后发送。')
    if result['kind'] == 'file':
        mime = FILE_TYPES.get(PurePosixPath(result['name']).suffix.lower())
        if mime is None:
            raise AttachmentError('暂不支持这个文件格式。请发送 PDF、Word、Excel、PowerPoint 或文本文件。')
        result['mime'] = mime
    return result


def qq_items(items):
    result = []
    if not isinstance(items, list):
        raise AttachmentError('附件消息格式无效，请重新发送。')
    for item in items:
        if not isinstance(item, dict):
            raise AttachmentError('附件消息格式无效，请重新发送。')
        mime = str(item.get('content_type') or '').lower()
        if mime == 'voice' or mime.startswith(('audio/', 'video/')):
            raise AttachmentError('当前支持图片和文件阅读，语音、视频暂未接入。')
        name = str(item.get('filename') or '')
        kind = 'image' if mime.startswith('image/') or PurePosixPath(name).suffix.lower() in (
            '.png', '.jpg', '.jpeg', '.webp', '.gif') else 'file'
        result.append(dict(kind=kind, platform='qq', url=item.get('url'), name=name,
                           mime=mime, size=item.get('size')))
    return result


def qq_media(data):
    current = qq_items(data.get('attachments') or [])
    quoted, ids = [], []
    reference = data.get('message_reference')
    if isinstance(reference, dict) and reference.get('message_id'):
        ids.append(str(reference['message_id']))
    # Tencent's current SDK: message_type=103 carries the quote in msg_elements.
    if data.get('message_type') == 103:
        for element in (data.get('msg_elements') or [])[:1]:
            if isinstance(element, dict):
                original = qq_items(element.get('attachments') or [])
                quoted.extend(original)
                if element.get('msg_idx') and (original or not element.get('content')):
                    ids.append(str(element['msg_idx']))
    return current, quoted, ids


def weixin_item(item):
    kind = 'image' if item.get('type') == 2 else 'file'
    content = item.get('image_item' if kind == 'image' else 'file_item') or {}
    if not isinstance(content, dict):
        raise AttachmentError('微信附件信息不完整，请重新发送。')
    media = content.get('media') or {}
    if not isinstance(media, dict):
        raise AttachmentError('微信附件信息不完整，请重新发送。')
    param = media.get('encrypt_query_param')
    url = media.get('full_url') or (
        'https://novac2c.cdn.weixin.qq.com/c2c/download?encrypted_query_param=' + quote(str(param), safe='')
        if param else '')
    key = media.get('aes_key') or ''
    if kind == 'image' and content.get('aeskey'):
        try:
            raw = bytes.fromhex(content['aeskey'])
            if len(raw) != 16:
                raise ValueError()
            key = base64.b64encode(raw).decode('ascii')
        except (ValueError, TypeError):
            raise AttachmentError('微信图片解密信息无效，请重新发送。') from None
    if kind == 'file' and not key:
        raise AttachmentError('微信文件缺少解密信息，请重新发送。')
    return dict(kind=kind, platform='weixin', url=url, key=key,
                name=content.get('file_name') or 'image',
                size=content.get('len') if kind == 'file' else content.get('mid_size'))


def weixin_media(msg):
    current, quoted, ids, texts = [], [], [], []
    items = msg.get('item_list') or []
    if not isinstance(items, list):
        raise AttachmentError('微信消息格式无效，请重新发送。')
    for item in items:
        if not isinstance(item, dict):
            raise AttachmentError('微信消息格式无效，请重新发送。')
        if item.get('type') == 1:
            content = item.get('text_item') or {}
            if not isinstance(content, dict) or not isinstance(content.get('text'), str):
                raise AttachmentError('微信文字消息格式无效。')
            texts.append(content['text'])
        elif item.get('type') in (2, 4):
            current.append(weixin_item(item))
        else:
            raise AttachmentError('当前支持文字、图片和文件，语音、视频暂未接入。')
        ref = item.get('ref_msg')
        if isinstance(ref, dict):
            original = ref.get('message_item')
            if isinstance(original, dict) and original.get('type') in (2, 4):
                quoted.append(weixin_item(original))
            elif ref.get('svr_id'):
                ids.append(str(ref['svr_id']))
    return '\n'.join(texts).strip(), current, quoted, ids


class AttachmentStore:
    """Short-lived pending references and a bounded explicit-reference index."""
    def __init__(self, db):
        self.db = db
        self.pending = {}
        db.execute('CREATE TABLE IF NOT EXISTS attachment_refs ('
                   'scope TEXT, sender TEXT, mid TEXT, items TEXT, at REAL, PRIMARY KEY(scope,sender,mid))')
        self.prune()

    def prune(self, now=None):
        now = time.time() if now is None else now
        self.pending = {key: value for key, value in self.pending.items() if now - value[0] < PENDING_SECONDS}
        self.db.execute('DELETE FROM attachment_refs WHERE at < ?', (now - REFERENCE_SECONDS,))
        self.db.execute('DELETE FROM attachment_refs WHERE rowid NOT IN '
                        '(SELECT rowid FROM attachment_refs ORDER BY at DESC, rowid DESC LIMIT 200)')
        self.db.commit()

    def clear(self, scope):
        self.pending = {key: value for key, value in self.pending.items() if key[0] != scope}
        self.db.execute('DELETE FROM attachment_refs WHERE scope=?', (scope,))
        self.db.commit()

    def discard(self, scope, sender):
        self.pending.pop((scope, sender), None)

    def prepare(self, scope, sender, mid, text, current=(), quoted=(), ids=(), now=None):
        now = time.time() if now is None else now
        self.prune(now)
        key = (scope, sender)
        previous = self.pending.pop(key, (0, []))[1]
        current = [validate(item) for item in bounded(current)]
        quoted = [validate(item) for item in bounded(quoted)]
        rows = self.db.execute('SELECT mid,items FROM attachment_refs WHERE scope=? AND sender=? '
                               'ORDER BY at DESC,rowid DESC', (scope, sender)).fetchall()
        if current:
            self.db.execute('INSERT OR REPLACE INTO attachment_refs VALUES (?,?,?,?,?)',
                            (scope, sender, str(mid), json.dumps(current, ensure_ascii=False), now))
            self.prune(now)
        explicit = bool(quoted)
        if not quoted and ids:
            quoted = [item for ref_id, payload in reversed(rows) if ref_id in ids for item in json.loads(payload)]
            explicit = bool(quoted)
            if not quoted:
                raise AttachmentError('找不到所引用的附件，可能已过期或不属于你的附件，请重新发送。')
        names_current = {item['name'] for item in current if item['name'] not in ('image', 'file')}
        if not explicit and text and not any(name in text for name in names_current):
            matches = []
            for _, payload in rows:
                items = json.loads(payload)
                matches.extend(item for item in items if item['name'] not in ('image', 'file')
                               and item['name'] in text)
            if matches:
                names = {item['name'] for item in matches}
                if any(len({item['url'] for item in matches if item['name'] == name}) > 1 for name in names):
                    raise AttachmentError('找到多个同名附件，请使用回复引用指定消息。')
                quoted, explicit = list(reversed(matches)), True
            else:
                ref = re.search(r'(?:上一|上|刚才(?:的)?|刚刚(?:的)?|之前(?:的)?|最近(?:的)?)(?:一|两|二|2)?(?:张|个|份)?(图片|照片|截图|图|文件|附件|文档|表格)', text)
                if ref:
                    kind = 'image' if ref[1] in ('图片', '照片', '截图', '图') else 'file'
                    candidates = [item for _, payload in rows for item in reversed(json.loads(payload)) if item['kind'] == kind]
                    count = 2 if kind == 'image' and re.search(r'两|二|2', ref[0]) else 1
                    if not candidates:
                        raise AttachmentError('找不到对应的最近附件，请重新发送或回复引用原消息。')
                    quoted, explicit = list(reversed(candidates[:count])), True
        # An explicit reference wins over unrelated pending/new files. New and
        # referenced images can be compared together within the two-image cap.
        chosen = bounded(current + quoted if explicit else previous + current)
        if not text and current and not explicit:
            self.pending[key] = (now, chosen)
            label = f"{len(chosen)}张图片" if chosen[0]['kind'] == 'image' else '1个文件'
            return [], f'已暂存最近{label}，请在5分钟内紧接着发送问题。每轮最多使用两张图片或一个文件。'
        return chosen, None


async def download(session, item, qq_token=None):
    """Fetch bounded bytes without sending credentials to redirects or other hosts."""
    url = safe_url(item['url'], item['platform'])
    for _ in range(4):
        headers = {}
        if item['platform'] == 'qq' and urlsplit(url).hostname == 'multimedia.nt.qq.com.cn' and qq_token:
            headers['Authorization'] = 'QQBot ' + await qq_token()
        try:
            async with session.get(url, headers=headers, allow_redirects=False,
                                   timeout=aiohttp.ClientTimeout(total=45), auto_decompress=False) as response:
                if response.status in (301, 302, 303, 307, 308):
                    url = safe_url(urljoin(url, response.headers.get('Location', '')), item['platform'])
                    continue
                if response.status != 200:
                    raise AttachmentError('附件下载失败或链接已过期，请重新发送原附件。')
                if response.content_length and response.content_length > MAX_BYTES + 16:
                    raise AttachmentError('单个图片或文件最多20 MiB，请压缩或拆分后发送。')
                data = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    data.extend(chunk)
                    if len(data) > MAX_BYTES + 16:
                        raise AttachmentError('单个图片或文件最多20 MiB，请压缩或拆分后发送。')
                return bytes(data)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            raise AttachmentError('附件下载连接失败，请重新发送后再试。') from None
    raise AttachmentError('附件下载重定向过多，请重新发送。')


def decrypt(data, key):
    try:
        raw = base64.b64decode(key, validate=True)
        if len(raw) == 32:
            raw = bytes.fromhex(raw.decode('ascii'))
        if len(raw) != 16:
            raise ValueError()
        decoder = Cipher(algorithms.AES(raw), modes.ECB()).decryptor()
        padded = decoder.update(data) + decoder.finalize()
        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(padded) + unpadder.finalize()
    except (ValueError, TypeError, UnicodeError, binascii.Error):
        raise AttachmentError('微信附件解密失败，请重新发送原附件。') from None


def image_mime(data):
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return 'image/gif'
    if data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return 'image/webp'
    raise AttachmentError('图片格式无法识别，请使用 PNG、JPEG、WEBP 或 GIF。')


async def input_parts(session, items, qq_token=None):
    result = []
    for item in bounded(items):
        item = validate(item)
        data = await download(session, item, qq_token)
        if item.get('key'):
            data = decrypt(data, item['key'])
        if not data or len(data) > MAX_BYTES:
            raise AttachmentError('附件为空或超过20 MiB，请重新发送。')
        mime = image_mime(data) if item['kind'] == 'image' else item['mime']
        uri = 'data:' + mime + ';base64,' + base64.b64encode(data).decode('ascii')
        result.append({'type': 'input_image', 'image_url': uri} if item['kind'] == 'image' else
                      {'type': 'input_file', 'filename': item['name'], 'file_data': uri})
    return result
