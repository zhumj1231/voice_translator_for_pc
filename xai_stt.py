"""xAI REST STT adapter, stdlib only. Credentials come only from XAI_API_KEY.

2026-09-19 official pricing: REST $0.10/audio hour; WebSocket $0.20/audio hour.
https://docs.x.ai/developers/models/speech-to-text
This application's cloud review uses bounded file clips, not a WebSocket stream.
"""
import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import http.client
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import ssl
import threading
from urllib.parse import urlsplit
import uuid

MODEL = 'grok-voice-transcribe-2.0'
ENDPOINT = 'https://api.x.ai/v1'
MAX_BYTES = 500_000_000


class RateLimited(RuntimeError):
    def __init__(self, retry_after=0):
        super().__init__('xAI HTTP 429：请求限流，请稍后再试')
        self.retry_after = retry_after


def api_key():
    return os.environ.get('XAI_API_KEY', '').strip()


def retry_delay(error, index):
    return max(2 ** (index + 1), error.retry_after)


def options(config):
    language = config.get('force_lang') or ''
    if language and not re.fullmatch(r'[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]{2,8})*', language):
        raise ValueError('force_lang 必须为语言代码，留空则自动识别')
    terms = config.get('keyterms', [])
    if not isinstance(terms, list) or len(terms) > 100 or any(
            not isinstance(t, str) or not 1 <= len(t) <= 50 or '\r' in t or '\n' in t for t in terms):
        raise ValueError('keyterms 最多 100 项，每项 1–50 字符')
    fields = [('model', config.get('model', MODEL)),
              ('format', 'true' if language and config.get('format', True) else 'false'),
              ('diarize', 'true' if config.get('diarization', True) else 'false'),
              ('filler_words', 'true' if config.get('filler_words', False) else 'false')]
    if language:
        fields.append(('language', language))
    fields.extend(('keyterm', t) for t in terms)
    return fields


def multipart(source, fields):
    """A fresh stream on every call: parameters first, file last; no 500 MB copies."""
    is_path = isinstance(source, Path)
    size = source.stat().st_size if is_path else len(source)
    if not 0 < size <= MAX_BYTES:
        raise ValueError('xAI 音频为空或超过 500 MB，请切片或压缩')
    suffix = source.suffix.lower() if is_path else '.wav'
    if not re.fullmatch(r'\.[a-z0-9]{1,8}', suffix):
        suffix = '.bin'
    filename = 'audio' + suffix
    mime = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
    boundary = 'XaiSubtitle' + uuid.uuid4().hex
    head = b''.join((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n').encode('utf-8')
                    for name, value in fields)
    head += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
             f'Content-Type: {mime}\r\n\r\n').encode('utf-8')
    tail = f'\r\n--{boundary}--\r\n'.encode('ascii')
    def chunks():
        yield head
        if is_path:
            with source.open('rb') as audio:
                while chunk := audio.read(65536):
                    yield chunk
        else:
            yield source
        yield tail
    return chunks(), {'Content-Type': 'multipart/form-data; boundary=' + boundary,
                      'Content-Length': str(len(head) + size + len(tail))}


def normalize(result):
    if not isinstance(result, dict) or not isinstance(result.get('text'), str):
        raise RuntimeError('xAI 响应缺少 text，保留本地结果')
    duration = result.get('duration')
    if isinstance(duration, bool) or not isinstance(duration, (float, int)) or not math.isfinite(duration) or duration < 0:
        raise RuntimeError('xAI 返回的音频时长无效')
    language = result.get('language', '')
    if not isinstance(language, str):
        raise RuntimeError('xAI 返回的语言字段无效')
    def words(items):
        if not isinstance(items, list):
            raise RuntimeError('xAI 返回的 words 无效')
        parsed = []
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get('text'), str):
                raise RuntimeError('xAI 逐词文本无效')
            start, end = item.get('start'), item.get('end')
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (start, end)) or not 0 <= start <= end <= duration + .25:
                raise RuntimeError('xAI 逐词时间戳超出音频范围')
            parsed.append({k:item[k] for k in ('text','start','end','speaker') if k in item})
        return parsed
    channels = result.get('channels', [])
    if not isinstance(channels, list):
        raise RuntimeError('xAI 返回的 channels 无效')
    parsed_channels = []
    for channel in channels:
        if not isinstance(channel, dict) or not isinstance(channel.get('text'), str):
            raise RuntimeError('xAI 声道结果无效')
        parsed_channels.append({'index':channel.get('index'), 'text':channel['text'],
                                'words':words(channel.get('words', []))})
    return {'text':result['text'].strip(), 'language':language, 'duration':duration,
            'words':words(result.get('words', [])), 'channels':parsed_channels}


def transcribe_once(source, config, duration_seconds=None):
    # Do not send a bearer credential to a configurable third-party endpoint.
    if config.get('endpoint', ENDPOINT).rstrip('/') != ENDPOINT:
        raise ValueError('xAI Key 只能发送至 https://api.x.ai/v1')
    key = api_key()
    if not key:
        raise RuntimeError('缺少环境变量 XAI_API_KEY；当前保留本地结果')
    body, headers = multipart(source, options(config))
    timeout = max(180, float(config.get('timeout_seconds', 180)), (duration_seconds or 0) * 2)
    proxy = config.get('proxy_url')
    context = ssl.create_default_context()
    if proxy:
        parsed = urlsplit(proxy)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost') or not parsed.port or parsed.username or parsed.password or parsed.path not in ('','/') or parsed.query or parsed.fragment:
            raise ValueError('xAI 仅接受本机 HTTP 代理地址，或留空直连')
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
        conn.set_tunnel('api.x.ai', 443)
    else:
        conn = http.client.HTTPSConnection('api.x.ai', timeout=timeout, context=context)
    try:
        conn.connect()
        if proxy:
            conn.sock = context.wrap_socket(conn.sock, server_hostname='api.x.ai')
        conn.request('POST', '/v1/stt', body=body,
                     headers={**headers, 'Authorization':'Bearer ' + key, 'Accept':'application/json'})
        response = conn.getresponse()
        if response.status == 429:
            value = response.getheader('Retry-After', '0')
            try:
                delay = float(value)
            except (ValueError, TypeError):
                try:
                    delay = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
                except (ValueError, TypeError):
                    delay = 0
            raise RateLimited(max(0, delay) if math.isfinite(delay) else 0)
        if response.status != 200:
            messages = {401:'密钥无效或未授权，请检查 XAI_API_KEY',
                        402:'余额不足，请到 console.x.ai 检查余额并充值',
                        413:'音频过大，请压缩或切片',
                        400:'请求参数或音频格式不被接受',
                        403:'访问被拒绝，请检查账户权限或网络出口',
                        503:'服务暂时不可用，保留本地结果'}
            # Avoid logging server echoes of credentials or uploaded audio.
            raise RuntimeError(f'xAI HTTP {response.status}：' + messages.get(response.status, '请求失败，保留本地结果'))
        data = response.read(32*1024*1024 + 1)
        if len(data) > 32*1024*1024:
            raise RuntimeError('xAI 响应过大，保留本地结果')
        result = normalize(json.loads(data))
        if duration_seconds is not None and abs(result['duration'] - duration_seconds) > .5:
            raise RuntimeError('xAI 返回时长与提交片段不符，保留本地结果')
        return result
    except ssl.SSLError:
        raise RuntimeError('xAI TLS 连接或证书验证失败；未重试') from None
    except TimeoutError:
        raise RuntimeError('xAI 请求超时；为避免重复计费未重试') from None
    except (OSError, http.client.HTTPException):
        raise RuntimeError('xAI 网络连接失败；保留本地结果，未重试') from None
    except (ValueError, UnicodeError):
        raise RuntimeError('xAI 响应格式异常；保留本地结果') from None
    finally:
        conn.close()


def transcribe_file(path, *, language=None, diarize=True, keyterms=None, proxy_url=None, max_retries=2):
    if not isinstance(max_retries, int) or not 0 <= max_retries <= 2:
        raise ValueError('max_retries 必须为 0、1 或 2')
    config = {'force_lang':language, 'diarization':diarize, 'keyterms':keyterms or [], 'proxy_url':proxy_url}
    for attempt in range(max_retries + 1):
        try:
            return transcribe_once(Path(path), config)
        except RateLimited as error:
            delay = retry_delay(error, attempt)
            if attempt >= max_retries or delay > 30:
                raise
            threading.Event().wait(delay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='读取本地音频，调用 xAI REST 转写并打印文本')
    parser.add_argument('audio', type=Path)
    parser.add_argument('--force-lang', default=None)
    parser.add_argument('--proxy', default=None, help='本机 HTTP 代理地址；省略则直连')
    parser.add_argument('--max-retries', type=int, choices=range(3), default=2)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    try:
        result = transcribe_file(args.audio, language=args.force_lang, proxy_url=args.proxy, max_retries=args.max_retries)
        print(json.dumps(result, ensure_ascii=False) if args.json else result['text'])
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(1, str(error) + '\n')
