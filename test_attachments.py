import asyncio
import base64
import contextlib
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from attachments import (AttachmentError, AttachmentStore, MAX_BYTES, REFERENCE_SECONDS,
                         bounded, decrypt, download, input_parts, qq_media, safe_url, weixin_media)
from bot import Bot
from test_personal import event
from test_weixin import message, record
from weixin import WeixinBot


PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')


def image(n=1):
    return dict(kind='image', platform='qq', url=f'https://gchat.qpic.cn/image{n}', name=f'pic{n}.png')


def document(n=1, name=None):
    return dict(kind='file', platform='qq', url=f'https://multimedia.nt.qq.com.cn/file{n}',
                name=name or f'report{n}.pdf')


def qq_attachment(item):
    return dict(url=item['url'], filename=item['name'],
                content_type='image/png' if item['kind'] == 'image' else 'file')


def wx_image(n=1):
    return {'type': 2, 'image_item': {'media': {'full_url': f'https://novac2c.cdn.weixin.qq.com/c2c/{n}'}}}


def wx_file(n=1):
    return {'type': 4, 'file_item': {'file_name': f'report{n}.pdf', 'media': {
        'encrypt_query_param': f'opaque-{n}', 'aes_key': base64.b64encode(b'k' * 16).decode()}}}


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.store = AttachmentStore(self.db)
        self.now = time.time()

    def tearDown(self):
        self.db.close()

    def prepare(self, text='', current=(), quoted=(), ids=(), **kw):
        return self.store.prepare(kw.pop('scope', 'chat'), kw.pop('sender', 'me'), kw.pop('mid', 'm'),
                                  text, current, quoted, ids, now=kw.pop('now', self.now))

    def test_three_separate_images_select_only_last_two_and_next_turn_has_none(self):
        for n in range(1, 4):
            items, ack = self.prepare(current=[image(n)], mid=str(n))
            self.assertEqual(items, [])
            self.assertIn('暂存', ack)
        chosen, _ = self.prepare('比较这两张')
        self.assertEqual([a['name'] for a in chosen], ['pic2.png', 'pic3.png'])
        self.assertEqual(self.prepare('继续')[0], [])

    def test_latest_kind_replaces_other_kind_and_only_latest_file_is_sent(self):
        self.prepare(current=[image(1), image(2)])
        self.prepare(current=[document(1), document(2)], mid='file')
        chosen, _ = self.prepare('总结内容')
        self.assertEqual([a['name'] for a in chosen], ['report2.pdf'])
        self.prepare(current=[document(3)], mid='file3')
        chosen, _ = self.prepare('描述', [image(3)])
        self.assertEqual([a['name'] for a in chosen], ['pic3.png'])

    def test_pending_expires_and_does_not_cross_user_or_conversation(self):
        self.prepare(current=[document()])
        self.assertEqual(self.prepare('问题', sender='other')[0], [])
        self.assertEqual(self.prepare('问题', scope='other')[0], [])
        self.assertEqual(self.prepare('问题', now=self.now + 301)[0], [])

    def test_reference_overrides_unrelated_pending_and_reuses_consumed_file(self):
        self.prepare('总结', [document()], mid='old')
        self.prepare(current=[image(2)], mid='new')
        chosen, _ = self.prepare('再检查这个文件', ids=['old'])
        self.assertEqual(chosen[0]['name'], 'report1.pdf')
        self.assertEqual(self.prepare('继续')[0], [])
        self.assertEqual(self.prepare('重新阅读 report1.pdf')[0][0]['name'], 'report1.pdf')

    def test_reference_wins_over_new_file_and_previous_image_excludes_current_image(self):
        self.prepare('看图', [image(1)], mid='i1')
        chosen, _ = self.prepare('和上一张图比较', [image(2)], mid='i2')
        self.assertEqual({item['name'] for item in chosen}, {'pic1.png', 'pic2.png'})
        self.prepare('读', [document(1)], mid='f1')
        chosen, _ = self.prepare('重读引用文件', [document(2)], ids=['f1'], mid='f2')
        self.assertEqual([item['name'] for item in chosen], ['report1.pdf'])
        chosen, _ = self.prepare('读 report1.pdf', [document(3, 'report1.pdf')], mid='f3')
        self.assertEqual(chosen[0]['url'], document(3)['url'])

    def test_natural_explicit_references_and_ambiguous_names(self):
        self.prepare('看图', [image(1), image(2)])
        self.assertEqual(len(self.prepare('重新比较刚才的两张图片')[0]), 2)
        self.assertEqual(self.prepare('上一张图有什么问题')[0][0]['name'], 'pic2.png')
        self.prepare('读', [document(1, 'same.pdf')], mid='f1')
        self.prepare('读', [document(2, 'same.pdf')], mid='f2')
        with self.assertRaisesRegex(AttachmentError, '同名'):
            self.prepare('总结 same.pdf')

    def test_expired_and_foreign_reference_never_fall_back_to_another_file(self):
        self.prepare('读', [document()], mid='private')
        with self.assertRaises(AttachmentError):
            self.prepare('读取这个文件', ids=['private'], sender='other')
        with self.assertRaises(AttachmentError):
            self.prepare('读取这个文件', ids=['private'], now=self.now + REFERENCE_SECONDS + 1)
        self.prepare(current=[document(2)])
        with self.assertRaises(AttachmentError):
            self.prepare('总结一下', ids=['missing'])
        self.assertEqual(self.prepare('继续')[0], [])

    def test_restart_keeps_only_explicit_index_and_clear_removes_it(self):
        self.prepare(current=[document()], mid='old')
        self.store = AttachmentStore(self.db)
        self.assertEqual(self.prepare('普通提问')[0], [])
        self.assertEqual(len(self.prepare('读取文件', ids=['old'])[0]), 1)
        self.store.clear('chat')
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM attachment_refs').fetchone()[0], 0)
        with self.assertRaises(AttachmentError):
            self.prepare('读取文件', ids=['old'])

    def test_reference_index_has_global_bound(self):
        for n in range(205):
            self.prepare('图', [image(n)], mid=str(n), now=self.now + n)
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM attachment_refs').fetchone()[0], 200)


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    def test_qq_current_and_embedded_quote_and_message_id(self):
        current, quoted, ids = qq_media({'attachments': [qq_attachment(image(2))],
            'message_type': 103, 'msg_elements': [{'msg_idx': 'old', 'attachments': [qq_attachment(document())]}]})
        self.assertEqual(current[0]['kind'], 'image')
        self.assertEqual(quoted[0]['name'], 'report1.pdf')
        self.assertEqual(ids, ['old'])

    def test_weixin_reference_content_and_reference_id(self):
        text, current, quoted, ids = weixin_media(message(item_list=[
            {'type': 1, 'text_item': {'text': '分析引用'}, 'ref_msg': {'message_item': wx_file()}},
            {**wx_image(), 'ref_msg': {'svr_id': 'prior'}}]))
        self.assertEqual(text, '分析引用')
        self.assertEqual(len(current), 1)
        self.assertEqual(quoted[0]['name'], 'report1.pdf')
        self.assertEqual(ids, ['prior'])

    async def test_weixin_keys_and_full_url_and_exact_model_input(self):
        key = b'k' * 16
        padder = padding.PKCS7(128).padder()
        padded = padder.update(PNG) + padder.finalize()
        encryptor = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
        encrypted = encryptor.update(padded) + encryptor.finalize()
        for encoded in (base64.b64encode(key), base64.b64encode(key.hex().encode())):
            self.assertEqual(decrypt(encrypted, encoded), PNG)
        item = wx_image()
        item['image_item'].update(aeskey=key.hex())
        item['image_item']['media']['aes_key'] = 'invalid-secondary-key'
        _, current, _, _ = weixin_media(message(item_list=[item]))
        with patch('attachments.download', AsyncMock(return_value=encrypted)):
            parts = await input_parts(None, current)
        self.assertEqual(parts, [{'type': 'input_image', 'image_url': 'data:image/png;base64,' + base64.b64encode(PNG).decode()}])
        with self.assertRaises(AttachmentError):
            decrypt(b'bad-ciphertext', base64.b64encode(key))

    async def test_file_input_uses_inline_data_and_filters_before_download(self):
        with patch('attachments.download', AsyncMock(return_value=b'%PDF-test')) as fetch:
            parts = await input_parts(None, [document(1), document(2)])
        self.assertEqual(fetch.await_count, 1)
        self.assertEqual(parts[0]['filename'], 'report2.pdf')
        self.assertEqual(parts[0]['type'], 'input_file')
        self.assertTrue(parts[0]['file_data'].startswith('data:application/pdf;base64,'))
        with patch('attachments.download', AsyncMock(return_value=PNG)) as fetch:
            self.assertEqual(len(await input_parts(None, [image(1), image(2), image(3)])), 2)
        self.assertEqual(fetch.await_count, 2)

    def test_untrusted_url_and_unsupported_media_fail_closed(self):
        for url in ('https://127.0.0.1/a', 'http://localhost/a', 'file:///private',
                    'https://qq.com.attacker.test/a', 'https://user:password@qq.com/a',
                    'https://qpic.cn:8443/a', 'https://qpic.cn/\nsecret'):
            with self.assertRaises(AttachmentError):
                safe_url(url, 'qq')
        with self.assertRaises(AttachmentError):
            weixin_media(message(item_list=[{'type': 3}]))

    async def test_size_format_and_invalid_images_fail_without_sending_model_input(self):
        for item in ({**image(), 'size': MAX_BYTES + 1}, document(name='program.exe')):
            with patch('attachments.download', AsyncMock()) as fetch:
                with self.assertRaises(AttachmentError):
                    await input_parts(None, [item])
                fetch.assert_not_awaited()
        with patch('attachments.download', AsyncMock(return_value=b'<html>expired</html>')):
            with self.assertRaises(AttachmentError):
                await input_parts(None, [image()])

    async def test_download_stream_limit_and_redirect_does_not_leak_token(self):
        class Response:
            status, content_length = 200, None
            headers = {}
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            @property
            def content(self):
                return self
            async def iter_chunked(self, size):
                yield b'x' * 20
                yield b'x' * 20
        session = MagicMock()
        session.get.return_value = Response()
        with patch('attachments.MAX_BYTES', 20):
            with self.assertRaises(AttachmentError):
                await download(session, image())
        redirect = Response()
        redirect.status = 302
        redirect.headers = {'Location': 'https://attacker.test/private'}
        session.get.reset_mock()
        session.get.return_value = redirect
        token = AsyncMock(return_value='private-token')
        with self.assertRaises(AttachmentError):
            await download(session, document(), token)
        self.assertEqual(session.get.call_count, 1)
        self.assertEqual(session.get.call_args.kwargs['headers'], {'Authorization': 'QQBot private-token'})


class QQAttachmentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.bot = Bot({'provider': 'chatgpt', 'data_dir': self.temp.name, 'owner_only': False}, ':memory:')
        self.bot.chatgpt.answer = AsyncMock(return_value=('分析完成', {}))
        self.bot.qq = AsyncMock(return_value={})
        self.downloader = patch('attachments.download', AsyncMock(return_value=PNG))
        self.fetch = self.downloader.start()

    def tearDown(self):
        self.downloader.stop()
        self.bot.db.close()
        self.temp.cleanup()

    async def drain(self):
        task = asyncio.create_task(self.bot.worker())
        try:
            await asyncio.wait_for(self.bot.queue.join(), 3)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def send(self, mid, text='', attachments=(), **extra):
        data = {**event(mid, text), 'attachments': [qq_attachment(a) for a in attachments], **extra}
        self.bot.accept('GROUP_AT_MESSAGE_CREATE', data)
        await self.drain()

    async def test_pending_selection_followup_and_explicit_quote_have_exact_uploads(self):
        for n in range(1, 4):
            await self.send(str(n), attachments=[image(n)])
        self.bot.chatgpt.answer.assert_not_awaited()
        self.fetch.assert_not_awaited()
        await self.send('question', '比较颜色')
        body = self.bot.chatgpt.answer.call_args.args[1]
        self.assertEqual(len(body[-1]['content']), 3)
        self.assertEqual(self.fetch.await_count, 2)
        await self.send('followup', '再解释一下')
        body = self.bot.chatgpt.answer.call_args.args[1]
        self.assertTrue(all(isinstance(m['content'], str) for m in body))
        self.assertEqual(self.fetch.await_count, 2)
        await self.send('quote', '检查这张图片', message_reference={'message_id': '1'})
        self.assertEqual(self.fetch.await_count, 3)
        self.assertEqual(self.fetch.call_args.args[1]['name'], 'pic1.png')
        saved = self.bot.db.execute('SELECT messages FROM group_memory').fetchone()[0]
        self.assertNotIn('base64', saved)
        self.assertNotIn('qpic.cn', saved)

    async def test_mixed_message_snapshot_cannot_acquire_later_attachment(self):
        for n in (1, 2):
            self.bot.accept('GROUP_AT_MESSAGE_CREATE', {**event(str(n), f'检查{n}'),
                'attachments': [qq_attachment(image(n))]})
        await self.drain()
        self.assertEqual([c.args[1]['name'] for c in self.fetch.await_args_list], ['pic1.png', 'pic2.png'])

    async def test_private_file_and_group_embedded_quote_use_inline_file_input(self):
        self.fetch.return_value = b'%PDF-test'
        self.bot.accept('C2C_MESSAGE_CREATE', {**event('private-file', '总结这个文件'),
            'attachments': [qq_attachment(document())]})
        await self.drain()
        part = self.bot.chatgpt.answer.call_args.args[1][-1]['content'][-1]
        self.assertEqual((part['type'], part['filename']), ('input_file', 'report1.pdf'))
        self.assertIn('/v2/users/self/messages', self.bot.qq.call_args.args[1])
        await self.send('embedded', '总结引用', message_type=103,
                        msg_elements=[{'msg_idx': 'other-message', 'attachments': [qq_attachment(document(2))]}])
        part = self.bot.chatgpt.answer.call_args.args[1][-1]['content'][-1]
        self.assertEqual(part['filename'], 'report2.pdf')

    async def test_responses_http_body_contains_only_current_attachment_parts(self):
        from test_chatgpt import Reply, sse
        from chatgpt import ChatGPT
        client = self.bot.chatgpt
        client.answer = ChatGPT.answer.__get__(client)
        client.resolve_model = AsyncMock(return_value={'slug': 'test-model'})
        client.access_token = AsyncMock(return_value='test-token')
        session = MagicMock()
        self.bot.http = session
        def response(*args, **kwargs):
            reply = Reply({})
            reply.content = sse([{'type': 'response.completed', 'response': {'output': [
                {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': '完成'}]}]}}])
            return reply
        session.post.side_effect = response
        await self.send('mixed', '分析', [image(1), image(2), image(3)])
        body = session.post.call_args.kwargs['json']
        self.assertEqual(len(body['input'][-1]['content']), 3)
        self.assertFalse(body['store'])
        await self.send('follow', '继续解释')
        body = session.post.call_args.kwargs['json']
        self.assertTrue(all(isinstance(m['content'], str) for m in body['input']))
        self.assertNotIn('base64', json.dumps(body))

    async def test_permissions_clear_and_failed_download(self):
        self.bot.c['owner_only'] = True
        await self.send('unauthorized', '看图', [image()])
        self.fetch.assert_not_awaited()
        self.assertEqual(self.bot.db.execute('SELECT COUNT(*) FROM attachment_refs').fetchone()[0], 0)
        self.bot.c['owner_only'] = False
        await self.send('pending', attachments=[image()])
        self.bot.clear_memory('group')
        await self.send('q', '看图')
        self.fetch.assert_not_awaited()
        self.fetch.side_effect = AttachmentError('附件下载失败，请重新发送。')
        self.bot.chatgpt.answer.reset_mock()
        await self.send('failed', '看图', [image()])
        self.bot.chatgpt.answer.assert_not_awaited()
        self.assertIn('下载失败', self.bot.qq.call_args.args[2]['content'])


class WeixinAttachmentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.bot = WeixinBot({'provider': 'chatgpt', 'data_dir': str(self.folder)}, self.folder, record())
        self.bot.chatgpt.answer = AsyncMock(return_value=('分析完成', {}))
        self.api = AsyncMock()
        self.downloader = patch('attachments.download', AsyncMock(return_value=PNG))
        self.fetch = self.downloader.start()

    def tearDown(self):
        self.downloader.stop()
        self.bot.db.close()
        self.temp.cleanup()

    async def send(self, mid, items):
        self.bot.ingest({'msgs': [message(mid, item_list=items)]})
        await self.bot.process_pending(self.api)

    async def test_images_and_quotes_work_and_history_never_reuploads_them(self):
        for n in range(1, 4):
            await self.send(str(n), [wx_image(n)])
        self.fetch.assert_not_awaited()
        await self.send('q', [{'type': 1, 'text_item': {'text': '比较颜色'}}])
        self.assertEqual(self.fetch.await_count, 2)
        self.assertEqual(len(self.bot.chatgpt.answer.call_args.args[1][-1]['content']), 3)
        await self.send('follow', [{'type': 1, 'text_item': {'text': '继续解释'}}])
        self.assertEqual(self.fetch.await_count, 2)
        self.assertTrue(all(isinstance(m['content'], str) for m in self.bot.chatgpt.answer.call_args.args[1]))
        await self.send('quote', [{'type': 1, 'text_item': {'text': '再看看图片'}, 'ref_msg': {'svr_id': '1'}}])
        self.assertEqual(self.fetch.await_count, 3)
        await self.send('inline', [{'type': 1, 'text_item': {'text': '分析'}, 'ref_msg': {'message_item': wx_image(9)}}])
        self.assertEqual(self.fetch.await_count, 4)
        self.assertNotIn('base64', self.bot.db.execute('SELECT messages FROM group_memory').fetchone()[0])
        self.assertEqual(self.bot.db.execute("SELECT COUNT(*) FROM wx_inbox WHERE payload != ''").fetchone()[0], 0)

    async def test_file_decryption_latest_only_and_explicit_filename(self):
        for n in (1, 2):
            await self.send(str(n), [wx_file(n)])
        with patch('attachments.decrypt', return_value=b'%PDF-test') as decode:
            await self.send('q', [{'type': 1, 'text_item': {'text': '总结'}}])
            self.assertEqual(decode.call_count, 1)
            part = self.bot.chatgpt.answer.call_args.args[1][-1]['content'][-1]
            self.assertEqual(part['filename'], 'report2.pdf')
            await self.send('ref', [{'type': 1, 'text_item': {'text': '再读 report1.pdf'}}])
            part = self.bot.chatgpt.answer.call_args.args[1][-1]['content'][-1]
            self.assertEqual(part['filename'], 'report1.pdf')

    async def test_failed_delivery_reuses_answer_without_uploading_again(self):
        self.api.send.side_effect = asyncio.TimeoutError
        with self.assertRaises(asyncio.TimeoutError):
            await self.send('mixed', [wx_image(), {'type': 1, 'text_item': {'text': '分析'}}])
        self.api.send.side_effect = None
        await self.bot.process_pending(self.api)
        self.assertEqual(self.fetch.await_count, 1)
        self.bot.chatgpt.answer.assert_awaited_once()
