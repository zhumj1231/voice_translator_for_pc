"""Local inference adapters. Only explicitly configured cloud adapters use the network."""
import base64
import audioop
import io
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import threading
import tempfile
import time
import urllib.request
import urllib.error
import urllib.parse
import wave
from audio_quality import audio_evidence
from gemini_credentials import get_key
from languages import native_language

from paths import ROOT, TOOLS, CRISPASR, LLAMA_SERVER, ASR_MODEL, TRANSLATION_MODEL, VAD_MODEL


def wav_bytes(pcm):
    output = io.BytesIO()
    with wave.open(output, 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(pcm)
    return output.getvalue()


def post_json(url, payload, headers=None, timeout=90):
    request = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                                    headers={'Content-Type': 'application/json', **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class LocalModels:
    def __init__(self, run_dir, status, stop):
        self.run_dir, self.status, self.stop = run_dir, status, stop
        self.processes = []
        self.logs = []
        self.lock = threading.Lock()
        self.asr_request_lock = threading.Lock()
        self.asr_url = self.translation_url = None

    def launch(self, args, name):
        if self.stop.is_set():
            raise InterruptedError('已停止')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        log = (self.run_dir / f'{name}.log').open('wb')
        with self.lock:
            if self.stop.is_set():
                log.close()
                raise InterruptedError('已停止')
            process = subprocess.Popen([str(a) for a in args] + ['--host', '127.0.0.1', '--port', str(port)],
                                       stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            self.processes.append(process)
            self.logs.append(log)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.stop.is_set():
                raise InterruptedError('已停止')
            if process.poll() is not None:
                raise RuntimeError(f'{name} 启动失败，见 {name}.log')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as response:
                    if json.load(response).get('status') == 'ok':
                        return f'http://127.0.0.1:{port}'
            except (OSError, ValueError):
                pass
            self.stop.wait(.25)
        raise TimeoutError(f'{name} 启动超时')

    def start(self, language, maximum=20):
        self.status('正在加载本地转写模型…')
        self.asr_url = self.launch([
            CRISPASR, '--server',
            '--backend', 'qwen3-1.7b', '-m', ASR_MODEL,
            '--no-auto-aligner', '--no-warmup', '--lid-backend', 'off', '-l', 'auto', '-t', '8'], 'asr')
        self.status('正在加载本地翻译模型…')
        self.translation_url = self.launch([
            LLAMA_SERVER, '-m', TRANSLATION_MODEL,
            '-c', '2048', '-ngl', 'auto', '--fit-target', '2048',
            '-b', '256', '-ub', '128', '-t', '8', '-fa', 'on', '-np', '1'], 'translation')
        self.translate('準備ができました。' if language == 'ja' else 'Ready.', [])
        # Preheat the finite ASR input shapes with both models resident, before capture starts.
        sizes = list(range(3, math.ceil(maximum/3)*3+1, 3))
        for index, seconds in enumerate(sizes, 1):
            if self.stop.is_set():
                raise InterruptedError('已停止')
            self.status(f'正在预热转写 {index}/{len(sizes)}；首次可能较久，完成后开始采集…')
            self.transcribe(bytes(seconds * 32000), language, vad=False, max_tokens=32)

    def detect_speech(self, pcm):
        # Standalone VAD avoids cropping away speech and changing the ASR graph shape.
        with tempfile.TemporaryDirectory(prefix='vad-', dir=self.run_dir) as folder:
            folder = Path(folder).resolve()
            assert folder.is_relative_to(self.run_dir.resolve())
            source, target = folder/'input.wav', folder/'vad.json'
            source.write_bytes(wav_bytes(pcm))
            args = [str(CRISPASR),
                    '-f', str(source), '--vad-export-raw', str(target),
                    '--vad-model', str(VAD_MODEL),
                    '-vt', '0.5', '-vspd', '150', '-vsd', '700', '-vp', '200', '-t', '4', '--no-gpu']
            with self.lock:
                if self.stop.is_set():
                    raise InterruptedError('已停止')
                process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
                self.processes.append(process)
            try:
                _, stderr = process.communicate(timeout=15)
                if process.returncode:
                    raise RuntimeError('语音检测失败：' + stderr.decode('utf-8', errors='replace')[-300:])
                return json.loads(target.read_text(encoding='utf-8'))
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=3)
                with self.lock:
                    if process in self.processes:
                        self.processes.remove(process)

    def transcribe(self, pcm, language, vad=True, max_tokens=220):
        # Each model process is private to this session. Serialize requests so the
        # diagnostic slice belongs to exactly this audio, including during warmup.
        with self.asr_request_lock:
            return self._transcribe(pcm, language, vad, max_tokens)

    def _transcribe(self, pcm, language, vad=True, max_tokens=220):
        detected = self.detect_speech(pcm) if vad else None
        evidence = audio_evidence(pcm)
        silent = bool(pcm) and audioop.rms(pcm, 2) <= 16
        tail_quiet = bool(pcm) and audioop.rms(pcm[-int(.65*32000):], 2) <= 32
        if vad and silent and not detected['crispasr_vad']['slices']:
            return {'text': '', 'segments': [], 'vad_segments': detected,
                    'audio_gate': {'silent': True, 'tail_quiet': True, **evidence}}
        # Zero padding uses no future audio. The speech detector's bounds remain unpadded.
        padded_samples = max(3*16000, math.ceil(len(pcm)/32000/3)*3*16000)
        pcm = pcm + bytes(max(0, padded_samples*2-len(pcm)))
        # Legacy callers may still pass ja/en. A forced decoder language can translate
        # speech into that language; captions must always preserve the spoken language.
        fields = {'language': 'auto', 'response_format': 'verbose_json', 'temperature': '0', 'seed': '42',
                  'max_tokens': str(max_tokens), 'no_auto_aligner': 'true', 'hotwords': '',
                  'vad': 'false', 'require_vad': 'false',
                  'vad_export': 'false', 'vad_export_raw': 'true',
                  'vad_threshold': '0.5', 'vad_min_silence_duration_ms': '700',
                  'vad_min_speech_duration_ms': '150', 'vad_speech_pad_ms': '200'}
        boundary = 'LocalSubtitleRequest'
        body = b''.join(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
                        for key, value in fields.items())
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="segment.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
        body += wav_bytes(pcm) + f'\r\n--{boundary}--\r\n'.encode()
        request = urllib.request.Request(self.asr_url + '/v1/audio/transcriptions', data=body,
                                        headers={'Content-Type': 'multipart/form-data; boundary=' + boundary})
        log_path = self.run_dir / 'asr.log'
        try:
            log_offset = log_path.stat().st_size
        except OSError:
            log_offset = None
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.load(response)
        result['detected_language'] = ''
        result['detected_language_source'] = ''
        if log_offset is not None and result.get('text', '').strip():
            try:
                with log_path.open('rb') as log:
                    log.seek(log_offset)
                    added = log.read(262145)
                if len(added) <= 262144:
                    result['detected_language'] = native_language(added.decode('utf-8', errors='replace'))
                    if result['detected_language']:
                        result['detected_language_source'] = 'qwen3'
            except OSError:
                pass  # A diagnostic failure must not break the transcription.
        if detected is not None:
            result['vad_segments'] = detected
        result['audio_gate'] = {'silent': silent, 'tail_quiet': tail_quiet, **evidence}
        return result

    def translate(self, text, context):
        # Context is separate data, never treated as another sentence to translate.
        prompt = ('将待译原文翻译为自然准确的简体中文，只输出该段译文。无论原文是什么语言，译文都用简体中文。前文仅帮助理解指代，'
                  '不要重复翻译前文，不要补写剧情，不要凭猜测改写人名，未说完的句子不要续写。\n'
                  + json.dumps({'前文': context[-2:], '待译原文': text}, ensure_ascii=False))
        result = post_json(self.translation_url + '/v1/chat/completions', {
            'messages': [{'role': 'user', 'content': prompt}], 'temperature': .1,
            'max_tokens': 512, 'stream': False})
        return result['choices'][0]['message']['content'].strip()

    def close(self):
        with self.lock:
            for process in self.processes:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            for log in self.logs:
                log.close()
            self.processes.clear()
            self.logs.clear()


class _NoCloudRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError('Gemini 请求出现重定向，已停止以保护凭据')


def gemini_post(url, payload, key, proxy_url=None):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != 'https' or parsed.netloc != 'generativelanguage.googleapis.com'
            or parsed.query or parsed.fragment):
        raise ValueError('Gemini 密钥只能发送至官方 HTTPS API')
    if proxy_url:
        from gemini_network import verified_post
        try:
            return verified_post(parsed.path, payload, key, proxy_url)
        except RuntimeError:
            raise
        except Exception:
            raise RuntimeError('Gemini 连接或响应异常，已停止；未重试') from None
    handlers = [_NoCloudRedirect()]
    request = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'x-goog-api-key': key})
    try:
        with urllib.request.build_opener(*handlers).open(request, timeout=45) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'Gemini HTTP {error.code}；未自动重试') from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError('Gemini 网络连接失败或超时；未自动重试') from None


class CloudReviewer:
    """One request per call; bounded 429 retries belong to the pipeline budget owner."""
    def __init__(self, config):
        self.config = config
        self.last_transcription = {}

    @property
    def available(self):
        if self.config.get('provider') == 'xai':
            from xai_stt import api_key
            return bool(self.config.get('enabled') and api_key())
        return bool(self.config.get('enabled') and self.config.get('model')
                    and self.config.get('endpoint') and get_key(self.config.get('key_env', 'GEMINI_API_KEY')))

    def review(self, pcm, language):
        cfg = self.config
        self.last_transcription = {}
        if cfg.get('provider') == 'xai':
            from xai_stt import transcribe_once
            if not pcm or len(pcm) % 2 or len(pcm) > 30 * 32000:
                raise ValueError('单次云端复核限 30 秒、16kHz 单声道 PCM')
            self.last_transcription = transcribe_once(wav_bytes(pcm), cfg, len(pcm)/32000)
            self.last_response = self.last_transcription
            return self.last_transcription['text']
        language_codes = []  # Detect the spoken language; never constrain it to a UI selection.
        key = get_key(cfg.get('key_env', 'GEMINI_API_KEY'))
        if not key:
            raise RuntimeError('尚未保存 Gemini API Key')
        if cfg['provider'] == 'gemini_interactions':
            if not pcm or len(pcm) % 2 or len(pcm) > 30 * 32000:
                raise ValueError('单次云端复核限 30 秒、16kHz 单声道 PCM')
            mode = {'type': 'verbatim'}
            if cfg.get('diarization'):
                mode['diarization_mode'] = 'speaker'
            payload = {'model': cfg['model'], 'store': False,
                       'input': [{'type': 'audio', 'mime_type': 'audio/wav',
                                  'data': base64.b64encode(wav_bytes(pcm)).decode('ascii')}],
                       'generation_config': {'max_output_tokens': 512, 'transcription_config': {
                           'language_codes': language_codes, 'mode': mode}}}
            result = gemini_post(cfg['endpoint'].rstrip('/') + '/interactions', payload, key,
                                 cfg.get('proxy_url'))
            self.last_response = result
            if result.get('status') != 'completed':
                raise RuntimeError('Gemini 没有返回完整转写；未自动重试')
            return ''.join(part.get('text', '') for step in result.get('steps', [])
                           if step.get('type') == 'model_output'
                           for part in step.get('content', []) if part.get('type') == 'text').strip()
        if cfg['provider'] != 'gemini_generate_content':
            raise ValueError('尚未配置支持的云端服务')
        endpoint = cfg['endpoint'].rstrip('/') + '/models/' + cfg['model'] + ':generateContent'
        prompt = ('只转写这段音频中真实说出的话语，自动识别原语言'
                  + '，保留原语言、语气词和未完成的句子。不要翻译、解释、编造背景或补全故事。'
                    '没有语音时输出空文本；无法听清的部分写[听不清]。')
        audio_part = {'inlineData': {'mimeType': 'audio/wav',
                                    'data': base64.b64encode(wav_bytes(pcm)).decode('ascii')}}
        if cfg['model'].startswith('gemini-3.5-transcribe'):
            payload = {'contents': [{'role': 'user', 'parts': [audio_part]}],
                       'generationConfig': {'audioTranscriptionConfig': {
                           'languageCodes': language_codes, 'mode': 'VERBATIM'}}}
        else:
            payload = {'contents': [{'role': 'user', 'parts': [{'text': prompt}, audio_part]}],
                       'generationConfig': {'temperature': 0, 'maxOutputTokens': 512}}
        result = gemini_post(endpoint, payload, key, cfg.get('proxy_url'))
        candidates = result.get('candidates', [])
        if not candidates or candidates[0].get('finishReason') not in (None, 'STOP'):
            raise RuntimeError('云端没有返回完整转写')
        return ''.join(part.get('text', '') for part in candidates[0].get('content', {}).get('parts', [])).strip()
