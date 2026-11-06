"""Windows loopback captions. Start does not touch or control the media player."""
import argparse
import copy
import audioop
from datetime import datetime
import json
import msvcrt
import os
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
import uuid
import wave

from backends import ROOT, CloudReviewer, LocalModels
from engine import Pipeline, RATE
from speakers import SpeakerClient, name as speaker_name

from languages import detected_language_label
from paths import DATA, config_path

def replay(path, audio, stop):
    with wave.open(str(path), 'rb') as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (RATE, 1, 2):
            raise ValueError('验证回放需要16kHz、单声道、16位PCM WAV')
        started, samples = time.monotonic(), 0
        while not stop.is_set():
            pcm = source.readframes(320)
            if not pcm:
                break
            samples += len(pcm)//2
            stop.wait(max(0, started+samples/RATE-time.monotonic()))
            if not stop.is_set():
                audio.append(pcm)


def loopback(audio, stop, emit):
    import pyaudiowpatch as pa
    with pa.PyAudio() as client:
        device = client.get_default_wasapi_loopback()
        rate, channels = int(device['defaultSampleRate']), device['maxInputChannels']
        state = None
        emit('status', '正在采集系统播放声音 · ' + device['name'])
        with client.open(format=pa.paInt16, channels=channels, rate=rate, input=True,
                         input_device_index=device['index'], frames_per_buffer=1024) as stream:
            while not stop.is_set():
                # Overflow is an error, rather than silently changing subtitle timing.
                pcm = stream.read(1024, exception_on_overflow=True)
                if channels == 2:
                    pcm = audioop.tomono(pcm, 2, .5, .5)
                elif channels > 2:
                    import numpy as np
                    frame = np.frombuffer(pcm, dtype='<i2').reshape(-1, channels)
                    pcm = frame.mean(axis=1).astype('<i2').tobytes()
                if rate != RATE:
                    pcm, state = audioop.ratecv(pcm, 2, 1, rate, RATE, state)
                audio.append(pcm)


class Session:
    def __init__(self, config, emit, replay_file=None):
        config = copy.deepcopy(config)
        config['language'] = 'auto'
        self.config, self.emit, self.replay_file = config, emit, replay_file
        self.stop = threading.Event()
        self.run_dir = DATA / 'sessions' / (datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:12])
        self.run_dir.mkdir(parents=True)
        self.models = LocalModels(self.run_dir, lambda text: emit('status', text), self.stop)
        self.pipeline = Pipeline(self.models, config, self.run_dir, emit, self.stop, CloudReviewer(config['cloud']))
        self.speakers = (SpeakerClient(config['speakers'], self.run_dir, self.stop)
                         if config.get('speakers',{}).get('enabled',False) else None)
        self.local_reviewer = (SpeakerClient({},self.run_dir,self.stop,
                                service='local_review_service.py',log_name='local-review')
                               if config.get('local_review',{}).get('enabled',False) else None)
        self.thread = None
        self.failed = False
        self.runtime_lock = None

    def acquire_runtime(self):
        handle = (DATA / '.runtime.lock').open('a+b')
        if handle.seek(0, 2) == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            handle.close()
            raise RuntimeError('已有一个字幕服务正在运行，请先停止它，避免争用显卡。') from None
        self.runtime_lock = handle

    def release_runtime(self):
        if self.runtime_lock is not None:
            self.runtime_lock.seek(0)
            msvcrt.locking(self.runtime_lock.fileno(), msvcrt.LK_UNLCK, 1)
            self.runtime_lock.close()
            self.runtime_lock = None

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def cancel(self):
        self.stop.set()
        self.pipeline.audio.finish()
        self.pipeline.translations.close(discard=True)
        self.pipeline.reviews.close(discard=True)
        self.pipeline.local_reviews.close(discard=True)
        # HTTP calls unblock when our own model processes stop.
        threading.Thread(target=self.models.close, daemon=True).start()
        if self.speakers:
            threading.Thread(target=self.speakers.close, daemon=True).start()
        if self.local_reviewer:
            threading.Thread(target=self.local_reviewer.close, daemon=True).start()

    def source_worker(self):
        try:
            if self.replay_file:
                self.emit('status', '正在按正常速度回放文件进行验证（不播放到扬声器）')
                replay(self.replay_file, self.pipeline.audio, self.stop)
            else:
                loopback(self.pipeline.audio, self.stop, self.emit)
        except Exception as error:
            self.failed = True
            self.emit('error', '声音采集停止：' + str(error))
            self.pipeline.event('capture_error', error=str(error))
        finally:
            self.pipeline.audio.finish()

    def run(self):
        source_thread = None
        try:
            self.acquire_runtime()
            if self.local_reviewer:
                self.emit('status','正在加载本地音频复核模型…')
                try:
                    self.local_reviewer.start()
                    self.pipeline.local_reviewer = self.local_reviewer
                    self.emit('quality_status','本地复核已启用 · 短词/喊声等弱结果先保留原文，复核后再翻译')
                except Exception as error:
                    self.local_reviewer.close()
                    self.pipeline.event('local_review_start_error',error=str(error))
                    self.emit('quality_status','本地复核未能启动，弱结果仍会标记待复核：'+str(error))
            if self.speakers:
                self.emit('status', '正在加载本地角色分析模型…')
                try:
                    self.speakers.start()
                    self.pipeline.speakers = self.speakers
                    self.emit('speaker_status', '角色区分：已启用 · A/B/C 为本次会话的声音分组')
                except Exception as error:
                    self.speakers.close()
                    self.pipeline.event('speaker_start_error', error=str(error))
                    self.emit('speaker_status', '角色区分未能启动，原文与翻译继续：'+str(error))
            self.models.start(self.config['language'], self.config['max_segment_seconds'])
            if self.stop.is_set():
                return
            self.pipeline.started = time.monotonic()
            (self.run_dir / 'settings.json').write_text(json.dumps(self.config, ensure_ascii=False, indent=2), encoding='utf-8')
            source_thread = threading.Thread(target=self.source_worker, daemon=True)
            source_thread.start()
            self.pipeline.run()
        except Exception as error:
            if not self.stop.is_set():
                self.failed = True
                self.emit('error', str(error))
                self.pipeline.event('session_error', error=str(error))
        finally:
            self.stop.set()
            self.pipeline.audio.finish()
            self.pipeline.translations.close(discard=True)
            self.pipeline.reviews.close(discard=True)
            self.pipeline.local_reviews.close(discard=True)
            self.models.close()
            if self.speakers:
                self.speakers.close()
            if self.local_reviewer:
                self.local_reviewer.close()
            if source_thread:
                source_thread.join(timeout=3)
            for worker in self.pipeline.threads:
                worker.join(timeout=1)
            self.pipeline.export()
            self.release_runtime()
            self.emit('finished', str(self.run_dir))


def load_config():
    config = json.loads(config_path().read_text(encoding='utf-8-sig'))
    # Migrate old fixed-language settings: transcription is always in the spoken language.
    config['language'] = 'auto'
    if not 1 <= config['refresh_seconds'] <= 10:
        raise ValueError('refresh_seconds 必须在1至10之间')
    if not 6 <= config['max_segment_seconds'] <= 30:
        raise ValueError('max_segment_seconds 必须在6至30之间')
    if config['refresh_seconds'] > config['max_segment_seconds']:
        raise ValueError('刷新间隔不能大于最长片段')
    speakers = config.get('speakers',{})
    if speakers.get('enabled') and not (1 <= speakers.get('refresh_seconds',3) <= 10
            and 10 <= speakers.get('window_seconds',15) <= 25
            and .2 <= speakers.get('match_threshold',.4) <= .9):
        raise ValueError('角色分析配置超出支持范围')
    return config


class Window:
    def __init__(self, replay_file=None):
        self.root = tk.Tk()
        self.root.title('本地双语字幕')
        self.root.geometry('1020x480')
        self.root.minsize(720, 340)
        self.root.configure(bg='#17202b')
        self.root.attributes('-topmost', True)
        self.updates = queue.Queue()
        self.session, self.replay_file = None, replay_file
        self.active_token = None
        self.last_error = ''
        self.config = load_config()
        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill='x')
        self.detected_language = tk.StringVar(value='原语言：待识别')
        ttk.Label(controls, textvariable=self.detected_language).pack(side='left', padx=6)
        ttk.Label(controls, text='译文：简体中文').pack(side='left', padx=6)
        self.vad = tk.BooleanVar(value=self.config['vad'])
        self.vad_box = ttk.Checkbutton(controls, text='语音检测', variable=self.vad)
        self.vad_box.pack(side='left', padx=6)
        self.speaker_enabled = tk.BooleanVar(value=self.config.get('speakers',{}).get('enabled',False))
        self.speaker_box = ttk.Checkbutton(controls, text='区分角色', variable=self.speaker_enabled)
        self.speaker_box.pack(side='left', padx=6)
        self.cloud_enabled = tk.BooleanVar(value=self.config['cloud'].get('enabled',False))
        self.cloud_box = ttk.Checkbutton(controls, text='云端复核', variable=self.cloud_enabled,
                                        command=self.update_cloud_label)
        self.cloud_box.pack(side='left', padx=6)
        self.start_button = ttk.Button(controls, text='开始', command=self.start)
        self.start_button.pack(side='left', padx=6)
        self.stop_button = ttk.Button(controls, text='停止', command=self.cancel, state='disabled')
        self.stop_button.pack(side='left', padx=6)
        self.record_dir = None
        self.record_button = ttk.Button(controls,text='查看记录',command=self.open_records)
        self.record_button.pack(side='left',padx=6)
        self.top = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text='置顶', variable=self.top,
                        command=lambda: self.root.attributes('-topmost', self.top.get())).pack(side='right')
        self.status = tk.StringVar(value='就绪。开始后先加载模型，再采集系统正在播放的声音。')
        self.metrics = tk.StringVar(value='')
        self.cloud_status = tk.StringVar()
        ttk.Label(self.root, textvariable=self.cloud_status, padding=(10, 3)).pack(fill='x')
        self.update_cloud_label()
        self.speaker_status = tk.StringVar(value='角色区分：声音分组仅供参考；名字暂用 A/B/C，不确定时保留待确认')
        ttk.Label(self.root, textvariable=self.speaker_status, padding=(10,3)).pack(fill='x')
        self.quality_status = tk.StringVar(value='疑难片段会保留原文和限量音频供复核，待复核不代表没有对白。')
        ttk.Label(self.root,textvariable=self.quality_status,wraplength=980,padding=(10,3)).pack(fill='x')
        ttk.Label(self.root, textvariable=self.status, wraplength=950, padding=(10, 4)).pack(fill='x')
        ttk.Label(self.root, textvariable=self.metrics, padding=(10, 2)).pack(fill='x')
        self.text = tk.Text(self.root, bg='#17202b', fg='#eaf0f7', wrap='word', font=('Microsoft YaHei UI', 13),
                            relief='flat', padx=15, pady=10, state='disabled')
        self.text.pack(fill='both', expand=True)
        self.text.tag_configure('meta', foreground='#8fc9cd', font=('Microsoft YaHei UI', 10))
        self.text.tag_configure('translation', foreground='#f7d98a')
        self.text.tag_configure('review', foreground='#bfc8e8', font=('Microsoft YaHei UI', 11))
        for index, color in enumerate(('#8bd3ff','#ffb6ce','#9ae4b7','#d2b3ff','#ffcb92','#ade4df')):
            self.text.tag_configure(f'speaker{index}', foreground=color, font=('Microsoft YaHei UI',13,'bold'))
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.after(100, self.poll)

    def update_cloud_label(self):
        if not self.cloud_enabled.get():
            self.cloud_status.set('云端复核：本次关闭')
            return
        config = {**self.config['cloud'], 'enabled': True}
        try:
            ready = CloudReviewer(config).available
        except OSError:
            ready = False
        mode = '疑难片段' if config.get('mode') != 'all' else '全部片段'
        provider = 'xAI Grok' if config.get('provider') == 'xai' else 'Google Gemini'
        self.cloud_status.set(
            f'{provider}：{mode}音频 · 每次开始最多 {config.get("max_audio_seconds_per_session",60)} 秒 / {config.get("max_requests_per_session",12)} 次 · 原文先显示，译文随后补充'
            if ready else f'{provider}：密钥未配置，当前只在本地处理')

    def start(self):
        if self.session is not None:
            return
        self.config['language'] = 'auto'
        self.config['vad'] = self.vad.get()
        self.config.setdefault('speakers',{})['enabled'] = self.speaker_enabled.get()
        self.config['cloud']['enabled'] = self.cloud_enabled.get()
        self.start_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.vad_box.configure(state='disabled')
        self.speaker_box.configure(state='disabled')
        self.cloud_box.configure(state='disabled')
        self.update_cloud_label()
        self.speaker_status.set('角色区分：等待加载' if self.speaker_enabled.get() else '角色区分：本次关闭')
        self.render([])
        token = self.active_token = object()
        self.last_error = ''
        self.session = Session(copy.deepcopy(self.config), lambda kind, value: self.updates.put((token, kind, value)), self.replay_file)
        self.record_dir = self.session.run_dir
        self.session.start()

    def open_records(self):
        folder = self.record_dir or DATA/'sessions'
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder))

    def cancel(self):
        if self.session:
            self.status.set('正在停止采集和模型…')
            self.stop_button.configure(state='disabled')
            self.session.cancel()

    def render(self, cues):
        self.detected_language.set('原语言：' + (detected_language_label(cues[-1]) if cues else '待识别'))
        self.text.configure(state='normal')
        self.text.delete('1.0', 'end')
        for cue in reversed(cues):
            marker = '本地整段稿' if cue['final'] else '本地暂定稿'
            self.text.insert('end', f"{cue['start']:.1f}–{cue['end']:.1f}秒 · {marker} · {cue['translation_status']} · 原语言：{detected_language_label(cue)}\n", 'meta')
            if self.speaker_enabled.get():
                speaker = cue.get('speaker')
                self.text.insert('end', '[' + speaker_name(speaker) + '] ',
                                 'meta' if speaker is None else f'speaker{speaker%6}')
                self.text.insert('end', cue.get('speaker_status','角色待确认')+'\n', 'meta')
            if cue.get('uncertain'):
                self.text.insert('end','原文待复核：未可靠检出对白，不据此生成中文结论\n','review')
            self.text.insert('end', '原文：' + (cue['original'] or '（未识别出文字）') + '\n')
            if cue['translation'] and (not cue.get('uncertain') or cue['translated_revision'] != cue['revision']):
                if cue['translated_revision'] != cue['revision']:
                    self.text.insert('end', '以下译文对应此前原文（不沿用当前角色标签）：' + cue['translated_from'] + '\n', 'meta')
                self.text.insert('end', '中文：' + cue['translation'] + '\n', 'translation')
            if cue.get('local_review_status'):
                self.text.insert('end',cue['local_review_status']+'\n','review')
            if cue.get('local_review_original') and cue['local_review_original'] != cue['original']:
                self.text.insert('end','本地复核候选：'+cue['local_review_original']+'\n','review')
                if cue.get('local_review_translation'):
                    self.text.insert('end','候选中文：'+cue['local_review_translation']+'\n','review')
            if cue.get('audio_clip'):
                self.text.insert('end','音频已保存：uncertain-audio/'+Path(cue['audio_clip']).name+'（查看记录）\n','meta')
            if cue['reasons']:
                self.text.insert('end', '待核对：' + '；'.join(cue['reasons']) + '\n', 'review')
            if cue['review_status']:
                timing = f" · 请求 {cue['review_seconds']:.1f}秒" if cue.get('review_seconds') else ''
                self.text.insert('end', cue['review_status'] + timing + '\n', 'review')
            if cue.get('review_language'):
                from languages import LANGUAGE_NAMES
                code = cue['review_language']
                label = LANGUAGE_NAMES.get(code.lower().split('-')[0], code)
                self.text.insert('end', '云端识别语种：' + label + '\n', 'meta')
            if cue['review_original'] and cue['review_original'] != cue['original']:
                self.text.insert('end', '云端原文：' + cue['review_original'] + '\n', 'review')
                if cue.get('review_translation'):
                    self.text.insert('end', '对应中文：' + cue['review_translation'] + '\n', 'review')
            self.text.insert('end', '\n')
        self.text.configure(state='disabled')

    def poll(self):
        try:
            while True:
                token, kind, value = self.updates.get_nowait()
                if token is not self.active_token:
                    continue
                if kind == 'captions':
                    self.render(value)
                elif kind == 'metrics':
                    self.metrics.set(value)
                elif kind == 'speaker_status':
                    self.speaker_status.set(value)
                elif kind == 'quality_status':
                    self.quality_status.set(value)
                elif kind == 'cloud_status':
                    self.cloud_status.set(value)
                elif kind == 'finished':
                    self.status.set(('错误：' + self.last_error + '；' if self.last_error else '') + '已停止。字幕与记录：' + value)
                    self.session = None
                    self.start_button.configure(state='normal')
                    self.stop_button.configure(state='disabled')
                    self.vad_box.configure(state='normal')
                    self.speaker_box.configure(state='normal')
                    self.cloud_box.configure(state='normal')
                else:
                    if kind == 'error':
                        self.last_error = value
                    self.status.set(('错误：' if kind == 'error' else '') + value)
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def close(self):
        if self.session:
            self.session.cancel()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--view-session', type=Path, help='只打开已保存字幕供查看，不自动开始采集')
    args = parser.parse_args()
    if args.headless:
        if not args.replay:
            parser.error('无界面验证必须指定 --replay，避免意外采集系统声音')
        def emit(kind, value):
            if kind != 'captions':
                print(json.dumps({'event': kind, 'value': value}, ensure_ascii=False), flush=True)
        session = Session(load_config(), emit, args.replay)
        session.run()
        if session.failed:
            raise SystemExit(1)
    else:
        window = Window(args.replay)
        if args.view_session:
            window.record_dir = args.view_session
            window.render(json.loads((args.view_session/'captions.json').read_text(encoding='utf-8')))
            window.status.set('正在查看保存记录，未采集声音。点击“开始”可新建实时会话。')
        try:
            window.root.mainloop()
        finally:
            if window.session:
                window.session.cancel()
                window.session.models.close()
                if window.session.thread:
                    window.session.thread.join(timeout=5)


if __name__ == '__main__':
    main()
