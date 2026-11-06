"""Continuous caption pipeline with bounded audio/work queues and revision-safe results."""
from collections import deque
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
import json
from pathlib import Path
import threading
import time
from speakers import SpeakerTimeline, name as speaker_name
from audio_quality import audio_evidence, content_key, weak_transcript
from backends import wav_bytes
from languages import LANGUAGE_NAMES, REVIEW_LANGUAGES
from xai_stt import RateLimited, retry_delay

RATE = 16000


class AudioBuffer:
    def __init__(self, seconds=60):
        self.limit = int(seconds * RATE) * 2
        self.data = bytearray()
        self.end = 0
        self.closed = False
        self.condition = threading.Condition()

    def append(self, pcm):
        if len(pcm) % 2:
            raise ValueError('PCM must contain complete 16-bit samples')
        with self.condition:
            if self.closed:
                return
            self.data.extend(pcm)
            self.end += len(pcm) // 2
            if len(self.data) > self.limit:
                del self.data[:len(self.data) - self.limit]
            self.condition.notify_all()

    def finish(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def read(self, start, end):
        with self.condition:
            first = self.end - len(self.data) // 2
            start, end = max(start, first), min(end, self.end)
            return start, end, bytes(self.data[(start-first)*2:(end-first)*2])

    def wait(self, after, step, stop):
        with self.condition:
            while not stop.is_set() and not self.closed and self.end < after + step:
                self.condition.wait(.1)
            return self.end, self.closed


class LatestJobs:
    """Retain at most one pending revision per cue, with a hard pending-count bound."""
    def __init__(self, capacity=2):
        self.capacity = capacity
        self.items = deque()
        self.closed = False
        self.condition = threading.Condition()

    def put(self, job):
        with self.condition:
            if self.closed:
                return [job]
            dropped = [old for old in self.items if old['id'] == job['id']]
            self.items = deque(old for old in self.items if old['id'] != job['id'])
            while len(self.items) >= self.capacity:
                dropped.append(self.items.popleft())
            self.items.append(job)
            self.condition.notify()
            return dropped

    def get(self):
        with self.condition:
            while not self.items and not self.closed:
                self.condition.wait()
            return self.items.popleft() if self.items else None

    def close(self, discard=False):
        with self.condition:
            self.closed = True
            if discard:
                self.items.clear()
            self.condition.notify_all()


@dataclass
class Cue:
    id: int
    start: float
    end: float
    original: str
    revision: int = 1
    final: bool = False
    translation: str = ''
    translated_from: str = ''
    translated_revision: int = 0
    translation_status: str = '等待句子稳定'
    review_status: str = ''
    review_original: str = ''
    review_translation: str = ''
    review_seconds: float = 0
    review_translation_seconds: float = 0
    review_language: str = ''
    review_duration: float = 0
    review_audio_start: float = 0
    review_words: list = field(default_factory=list)
    review_channels: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    asr_seconds: float = 0
    speaker: int | None = None
    speaker_status: str = '角色待确认'
    uncertain: bool = False
    local_review_status: str = ''
    local_review_original: str = ''
    local_review_translation: str = ''
    audio_clip: str = ''
    detected_language: str = ''
    detected_language_source: str = ''


class CaptionStore:
    def __init__(self):
        self.lock = threading.RLock()
        self.cues = {}

    def snapshot(self, limit=None):
        with self.lock:
            values = list(self.cues.values())
            return [asdict(cue) for cue in (values[-limit:] if limit else values)]

    def translated(self, cue_id, revision, text, original=None):
        with self.lock:
            cue = self.cues[cue_id]
            if cue.uncertain and revision == cue.revision:
                return False
            if revision > cue.revision or revision <= cue.translated_revision:
                return False
            if revision != cue.revision and original is None:
                return False  # An older translation is usable only with its exact source.
            cue.translation = text
            cue.translated_from = original if original is not None else cue.original
            cue.translated_revision = revision
            cue.translation_status = ('此前原文译文' if revision != cue.revision else
                                      '已翻译' if cue.final else '暂译')
            return True

    def context(self, before):
        with self.lock:
            # Only finished preceding cues, never later audio or unpaired translations.
            prior = [cue for cue in self.cues.values()
                     if cue.id < before and cue.final and cue.original and not cue.uncertain][-2:]
            return [({'说话人': speaker_name(cue.speaker)} if cue.speaker is not None else {}) |
                    {'原文': cue.original[:200], '译文': cue.translation[:120]
                     if cue.translated_revision == cue.revision else ''} for cue in prior]


def vad_bounds(response):
    """Read the installed CrispASR VAD schema; absent metadata is not 'no speech'."""
    body = response.get('vad_segments', {}).get('crispasr_vad')
    if body is None:
        return None
    sample_rate = body['sample_rate']
    ranges = []
    for item in body['slices']:
        ranges.append((item['start'] / sample_rate, item['end'] / sample_rate))
    return ranges


class Pipeline:
    def __init__(self, models, config, run_dir, emit, stop, cloud=None, speakers=None, local_reviewer=None):
        self.models, self.config, self.run_dir = models, config, Path(run_dir)
        self.emit, self.stop, self.cloud = emit, stop, cloud
        self.audio = AudioBuffer()
        self.store = CaptionStore()
        self.translations, self.reviews = LatestJobs(2), LatestJobs(1)
        self.log_lock = threading.Lock()
        self.review_seconds = 0
        self.review_requests = 0
        self.review_failures = 0
        self.review_last_error = ''
        self.started = time.monotonic()
        self.threads = []
        self.speakers = speakers
        self.speaker_timeline = SpeakerTimeline()
        self.local_reviewer = local_reviewer
        self.local_reviews = LatestJobs(1)
        self.saved_audio_seconds = 0.
        self.archived_revisions = set()
        self.local_review_thread = None

    def archive_uncertain(self, cue, pcm, start):
        cfg = self.config.get('diagnostics',{})
        if not cfg.get('save_uncertain_audio',False) or not cue.reasons:
            return
        if (cue.id, cue.revision) in self.archived_revisions:
            return
        duration = len(pcm)/32000
        if self.saved_audio_seconds+duration > cfg.get('max_audio_seconds',120):
            self.event('diagnostic_audio_limit', id=cue.id)
            return
        folder = self.run_dir/'uncertain-audio'
        folder.mkdir(exist_ok=True)
        path = folder/f'cue-{cue.id:04}-r{cue.revision}.wav'
        path.write_bytes(wav_bytes(pcm))
        path.with_suffix('.json').write_text(json.dumps({'audio_start':start,'audio_end':start+duration,
            'original':cue.original,'reasons':cue.reasons,'sample_rate':RATE},ensure_ascii=False,indent=2),encoding='utf-8')
        self.saved_audio_seconds += duration
        self.archived_revisions.add((cue.id, cue.revision))
        cue.audio_clip = str(path)
        self.event('diagnostic_audio',id=cue.id,revision=cue.revision,path=str(path),
                   duration=duration,reasons=cue.reasons)

    def queue_local_review(self, cue, pcm):
        if not self.local_reviewer:
            return
        if cue.detected_language and cue.detected_language not in REVIEW_LANGUAGES:
            cue.local_review_status = '本地复核不支持' + LANGUAGE_NAMES.get(cue.detected_language, cue.detected_language) + '，保留主转写'
            cue.local_review_original = cue.local_review_translation = ''
            return
        cue.local_review_status = '本地另一模型复核中'
        for old in self.local_reviews.put({'id':cue.id,'revision':cue.revision,
                                          'original':cue.original,'pcm':pcm}):
            if old['id'] != cue.id:
                self.store.cues[old['id']].local_review_status = '本地复核积压，已跳过'
            self.event('local_review_superseded',id=old['id'],revision=old['revision'])

    def local_review_worker(self):
        failures = 0
        try:
            while (job := self.local_reviews.get()) is not None and not self.stop.is_set():
                if failures >= 2:
                    with self.store.lock:
                        self.store.cues[job['id']].local_review_status = '本地复核连续失败，已暂停'
                    self.changed()
                    continue
                try:
                    result = self.local_reviewer.analyze(job['pcm'],0)
                    if self.stop.is_set():
                        break
                    candidate = result.get('text','').strip()
                    speech = result.get('audio_event') == '<|Speech|>'
                    detected_language = result.get('language')
                    known_language = detected_language in ('<|ja|>', '<|en|>', '<|zh|>', '<|ko|>', '<|yue|>')
                    # The event classifier can call speech-over-music "BGM".
                    # Keep substantial recognized-language text as a labelled candidate.
                    usable = bool(content_key(candidate)) and (speech or (known_language and
                        not weak_transcript(candidate, [], max(2,len(job['pcm'])/32000))))
                    with self.store.lock:
                        cue = self.store.cues[job['id']]
                        if cue.revision != job['revision']:
                            self.event('local_review_stale',id=cue.id,revision=job['revision'])
                            continue
                        # A five-language secondary recognizer is not a universal LID.
                        # Never let its language prediction replace the primary result.
                        conflict = (cue.detected_language and detected_language != '<|' + cue.detected_language + '|>')
                        if conflict:
                            cue.local_review_original = cue.local_review_translation = ''
                            cue.local_review_status = '复核语种与主转写不一致，未采用复核候选'
                            self.event('local_review_language_conflict', id=cue.id, revision=cue.revision,
                                       primary_language=cue.detected_language, **result)
                            self.changed()
                            continue
                        cue.local_review_original = candidate if usable else ''
                        cue.local_review_translation = ''
                        if usable:
                            same = content_key(candidate) == content_key(cue.original)
                            cue.local_review_status = (
                                '复核检出文字但声音类型判为背景声，保留候选供核对' if not speech else
                                '本地两模型转写一致，仍可能有误' if same and known_language else
                                '文字相同但语言未可靠识别，仍待复核' if same else
                                '本地两模型转写不同，保留候选供核对')
                            if same and known_language and speech and cue.uncertain:
                                cue.uncertain = False
                                self.queue_translation(cue)
                        else:
                            same = False
                            cue.local_review_status = '本地复核未检出可靠对白；可能是音乐、音效或语音漏检'
                        self.event('local_review',id=cue.id,revision=cue.revision,
                                   original=job['original'],**result)
                    self.changed()
                    if usable and known_language and not same:
                        translated = self.models.translate(candidate,self.store.context(job['id']))
                        with self.store.lock:
                            if not self.stop.is_set() and cue.revision == job['revision']:
                                cue.local_review_translation = translated
                except Exception as error:
                    failures += 1
                    with self.store.lock:
                        self.store.cues[job['id']].local_review_status = '本地复核失败，保留原文'
                    self.event('local_review_error',id=job['id'],error=str(error))
                self.changed()
        finally:
            self.local_reviewer.close()

    def speaker_worker(self):
        last = 0
        cfg = self.config.get('speakers', {})
        try:
            while not self.stop.is_set():
                available, closed = self.audio.wait(last, round(cfg.get('refresh_seconds',3)*RATE), self.stop)
                if self.stop.is_set() or (closed and available <= last):
                    break
                start, end, pcm = self.audio.read(max(0, available-round(cfg.get('window_seconds',15)*RATE)), available)
                result = self.speakers.analyze(pcm, start/RATE, closed)
                if self.stop.is_set():
                    break
                last = end
                self.speaker_timeline.update(result)
                self.event('speakers', **result)
                with self.store.lock:
                    for cue in self.store.cues.values():
                        if cue.end >= start/RATE:
                            cue.speaker, cue.speaker_status = self.speaker_timeline.describe(cue.start, cue.end)
                self.changed()
        except Exception as error:
            if not self.stop.is_set():
                self.event('speaker_error', error=str(error))
                self.speaker_timeline.status = '角色分析失败'
                self.emit('speaker_status', '角色分析失败，原文与翻译继续；可停止后重试')
                with self.store.lock:
                    for cue in self.store.cues.values():
                        if cue.speaker is None:
                            cue.speaker_status = '角色分析失败'
                self.changed()
        finally:
            self.speakers.close()

    def event(self, kind, **values):
        record = {'event': kind, 'wall_seconds': round(time.monotonic()-self.started, 3), **values}
        with self.log_lock:
            with (self.run_dir / 'events.jsonl').open('a', encoding='utf-8') as output:
                output.write(json.dumps(record, ensure_ascii=False) + '\n')

    def changed(self):
        self.emit('captions', self.store.snapshot(limit=20))

    def queue_translation(self, cue):
        if cue.uncertain:
            cue.translation_status = '原文待复核，暂缓翻译'
            return
        cue.translation_status = '翻译中' if cue.final else '暂译中'
        job = {'id': cue.id, 'revision': cue.revision, 'original': cue.original}
        for old in self.translations.put(job):
            with self.store.lock:
                dropped = self.store.cues[old['id']]
                if dropped.revision == old['revision'] and dropped.id != cue.id:
                    dropped.translation_status = '积压：已跳过翻译'
            self.event('translation_job_superseded', id=old['id'], revision=old['revision'])

    def translation_worker(self):
        while (job := self.translations.get()) is not None and not self.stop.is_set():
            with self.store.lock:
                cue = self.store.cues[job['id']]
                if cue.revision != job['revision'] or cue.uncertain:
                    continue
            started = time.monotonic()
            try:
                translated = self.models.translate(job['original'], self.store.context(job['id']))
                if self.stop.is_set():
                    break
                accepted = self.store.translated(job['id'], job['revision'], translated, job['original'])
                self.event('translation', id=job['id'], revision=job['revision'], accepted=accepted,
                           matches_current=self.store.cues[job['id']].revision == job['revision'],
                           seconds=round(time.monotonic()-started, 3))
            except Exception as error:
                with self.store.lock:
                    cue = self.store.cues[job['id']]
                    if cue.revision == job['revision']:
                        cue.translation_status = '翻译失败，保留原文'
                self.event('translation_error', id=job['id'], error=str(error))
            self.changed()

    def cloud_pause_status(self):
        return '云端已暂停（连续失败）：' + (self.review_last_error or '原因未记录，请查看记录')

    def queue_review(self, cue, pcm, audio_start=None):
        if not self.cloud or not self.cloud.available:
            return
        if not cue.reasons and self.config['cloud'].get('mode', 'uncertain') != 'all':
            return
        duration = len(pcm) / 32000
        if self.review_requests >= self.config['cloud'].get('max_requests_per_session', 12):
            cue.review_status = '达到本次云端请求上限'
            return
        if self.review_failures >= 2:
            cue.review_status = self.cloud_pause_status()
            return
        if self.review_seconds + duration > self.config['cloud'].get('max_audio_seconds_per_session', 300):
            cue.review_status = '达到本次云端音频上限'
            return
        cue.review_status = '等待云端复核'
        for old in self.reviews.put({'id': cue.id, 'revision': cue.revision, 'pcm': pcm,
                                    'audio_start':cue.start if audio_start is None else audio_start,
                                    'queued_at': time.monotonic()}):
            with self.store.lock:
                self.store.cues[old['id']].review_status = '复核积压：已跳过'

    def request_cloud(self, job, duration):
        retries = min(2, max(0, int(self.config['cloud'].get('max_429_retries', 2))))
        for attempt in range(retries + 1):
            with self.store.lock:
                if self.stop.is_set() or self.store.cues[job['id']].revision != job['revision']:
                    raise InterruptedError('复核已取消或原文已更新')
            if self.review_requests >= self.config['cloud'].get('max_requests_per_session', 12):
                raise RuntimeError('达到本次云端请求上限，未继续重试')
            if self.review_seconds + duration > self.config['cloud'].get('max_audio_seconds_per_session', 300):
                raise RuntimeError('达到本次云端音频上限，未继续重试')
            self.review_seconds += duration
            self.review_requests += 1
            self.emit('cloud_status', f'云端预算已占用 {self.review_seconds:.1f}/{self.config["cloud"].get("max_audio_seconds_per_session",300)} 秒音频 · {self.review_requests}/{self.config["cloud"].get("max_requests_per_session",12)} 次尝试（含失败）')
            try:
                return self.cloud.review(job['pcm'], self.config['language'])
            except RateLimited as error:
                delay = retry_delay(error, attempt)
                if self.config['cloud'].get('provider') != 'xai' or attempt >= retries or delay > 30:
                    raise
                self.event('cloud_rate_limited', id=job['id'], wait_seconds=delay, attempt=attempt+1)
                self.emit('cloud_status', f'xAI 限流，{delay:g} 秒后重试；重试仍受本次额度限制')
                if self.stop.wait(delay):
                    raise InterruptedError('复核已停止') from None

    def review_worker(self):
        while (job := self.reviews.get()) is not None and not self.stop.is_set():
            duration = len(job['pcm']) / 32000
            with self.store.lock:
                if self.store.cues[job['id']].revision != job['revision']:
                    continue
            if self.review_requests >= self.config['cloud'].get('max_requests_per_session', 12):
                with self.store.lock:
                    self.store.cues[job['id']].review_status = '达到本次云端请求上限'
                self.changed()
                continue
            if self.review_seconds + duration > self.config['cloud'].get('max_audio_seconds_per_session', 300):
                with self.store.lock:
                    self.store.cues[job['id']].review_status = '达到本次云端音频上限'
                self.changed()
                continue
            if self.review_failures >= 2:
                with self.store.lock:
                    self.store.cues[job['id']].review_status = self.cloud_pause_status()
                self.changed()
                continue
            try:
                started = time.monotonic()
                result = self.request_cloud(job, duration)
                elapsed = time.monotonic()-started
                self.review_failures = 0
                self.review_last_error = ''
                if self.stop.is_set():
                    break
                with self.store.lock:
                    cue = self.store.cues[job['id']]
                    if cue.revision != job['revision']:
                        continue
                    # Keep both sources. Different is not evidence that cloud is correct.
                    cue.review_original = result
                    cue.review_translation = ''
                    metadata = getattr(self.cloud, 'last_transcription', {})
                    cue.review_language = metadata.get('language', '')
                    cue.review_duration = metadata.get('duration', 0)
                    cue.review_audio_start = job.get('audio_start', cue.start)
                    # Word timestamps are relative to the uploaded PCM, not the VAD
                    # speech start. Speaker numbers are clip-local, not session A/B/C.
                    def absolute_words(words):
                        return [{**w, 'start':w['start']+cue.review_audio_start,
                                 'end':w['end']+cue.review_audio_start} for w in words]
                    cue.review_words = absolute_words(metadata.get('words', []))
                    cue.review_channels = [{**c, 'words':absolute_words(c.get('words', []))}
                                           for c in metadata.get('channels', [])]
                    cue.review_seconds = round(elapsed, 3)
                    cue.review_status = ('云端未识别出文字，保留本地结果' if not result else
                                         '云端与本地一致' if result == cue.original else '云端有不同转写，供对照')
                # Publish source immediately, before local Chinese translation starts.
                self.changed()
                self.event('cloud_review', id=job['id'], sent_audio_seconds=duration,
                           provider=self.config['cloud']['provider'], different=result != cue.original,
                           seconds=round(elapsed,3), empty=not bool(result),
                           queue_seconds=round(started-job.get('queued_at',started),3),
                           usage=getattr(self.cloud,'last_response',{}).get('usage'))
                if result and result != cue.original:
                    try:
                        translation_started = time.monotonic()
                        translation = self.models.translate(result, self.store.context(job['id']))
                        with self.store.lock:
                            if not self.stop.is_set() and cue.revision == job['revision']:
                                cue.review_translation = translation
                                cue.review_translation_seconds = round(time.monotonic()-translation_started,3)
                        self.event('cloud_translation', id=job['id'], revision=job['revision'],
                                   seconds=round(time.monotonic()-translation_started,3))
                    except Exception as error:
                        self.event('cloud_result_translation_error', id=job['id'], error=str(error))
            except InterruptedError:
                continue
            except Exception as error:
                self.review_failures += 1
                self.review_last_error = str(error)
                with self.store.lock:
                    self.store.cues[job['id']].review_status = (self.cloud_pause_status() if self.review_failures >= 2 else
                        '云端失败，继续使用本地结果：' + self.review_last_error)
                if self.review_failures >= 2:
                    self.emit('cloud_status', self.cloud_pause_status())
                self.event('cloud_error', id=job['id'], error=str(error))
            self.changed()

    def run(self):
        workers = [self.translation_worker, self.review_worker]
        if self.speakers:
            workers.append(self.speaker_worker)
        if self.local_reviewer:
            workers.append(self.local_review_worker)
        for worker in workers:
            thread = threading.Thread(target=worker, daemon=True)
            self.threads.append(thread)
            if worker == self.local_review_worker:
                self.local_review_thread = thread
            thread.start()
        cursor = last_end = 0
        current_id = None
        last_translation_end = 0
        last_local_review_end = 0
        interval = round(self.config['refresh_seconds'] * RATE)
        maximum = round(self.config['max_segment_seconds'] * RATE)
        try:
            while not self.stop.is_set():
                available, closed = self.audio.wait(last_end, interval, self.stop)
                if self.stop.is_set() or (closed and available <= last_end):
                    break
                # No FIFO of missed ticks. At most one ASR call is in flight.
                if available - cursor > maximum + 8 * RATE:
                    new_cursor = available - maximum
                    self.event('audio_gap', start=cursor/RATE, end=new_cursor/RATE,
                               reason='识别落后，转到最近音频')
                    with self.store.lock:
                        if current_id is not None:
                            cue = self.store.cues[current_id]
                            cue.final = True
                            cue.reasons.append('处理落后，片段不完整')
                            self.queue_translation(cue)
                    current_id = None
                    cursor = new_cursor
                    self.emit('status', '处理落后，已跳过部分音频；缺口已记入日志')
                planned_end = min(available, cursor+maximum)
                boundary = (self.speaker_timeline.boundary(cursor/RATE, planned_end/RATE)
                            if self.speakers else None)
                if boundary is not None:
                    planned_end = min(planned_end, round(boundary*RATE))
                start, end, pcm = self.audio.read(cursor, planned_end)
                if not pcm:
                    last_end = available
                    continue
                started = time.monotonic()
                response = self.models.transcribe(pcm, self.config['language'], vad=self.config['vad'])
                seconds = time.monotonic() - started
                if self.stop.is_set():
                    break
                ranges = vad_bounds(response) if self.config['vad'] else None
                if self.config['vad'] and ranges is None:
                    raise RuntimeError('语音检测没有返回时间边界，已停止以免静默降级')
                text = response.get('text', '').strip()
                evidence = {**audio_evidence(pcm), **response.get('audio_gate',{})}
                silent = evidence.get('silent',evidence.get('rms',1)<=16)
                if ranges == [] and response.get('audio_gate', {}).get('silent') is True:
                    text = ''
                length = (end-start) / RATE
                quiet_tail = bool(ranges and length-max(r[1] for r in ranges) >= .65
                                  and response.get('audio_gate', {}).get('tail_quiet', False))
                final = boundary is not None or quiet_tail or end-start >= maximum or (closed and end >= available)
                # The CPU worker may discover a change while ASR is running.
                # Keep this revision provisional and immediately re-cut the audio.
                discovered = (self.speaker_timeline.boundary(start/RATE,end/RATE)
                              if self.speakers and boundary is None else None)
                if discovered is not None:
                    final = False
                self.event('asr', start=start/RATE, end=end/RATE, text=text,
                           seconds=round(seconds, 3), final=final, vad=ranges,
                           detected_language=response.get('detected_language', ''),
                           detected_language_source=response.get('detected_language_source', ''),
                           audio=evidence,
                           backlog_seconds=round(max(0, self.audio.end-end)/RATE, 3))
                last_end = round(discovered*RATE) if discovered is not None else end
                if not text:
                    if not silent and not final:
                        self.event('asr_empty_retained',start=start/RATE,end=end/RATE,
                                   reason='有声音但暂无文字，等待更多上下文')
                        self.emit('quality_status','有声音但暂未识别出文字，正在保留上下文继续尝试')
                        continue
                    self.emit('quality_status','')
                    with self.store.lock:
                        if not silent and current_id is None:
                            current_id = max(self.store.cues,default=0)+1
                            self.store.cues[current_id] = Cue(current_id,start/RATE,end/RATE,'',
                                final=True,uncertain=True,reasons=['有声音但未识别出文字'])
                        if current_id is not None:
                            cue = self.store.cues[current_id]
                            cue.final = True
                            if cue.original:
                                cue.reasons.append('后续识别为空，保留此前暂定稿')
                            if not silent:
                                cue.uncertain = True
                                cue.end = end/RATE
                            self.queue_translation(cue)
                            if not silent:
                                self.archive_uncertain(cue,pcm,start/RATE)
                                self.queue_local_review(cue,pcm)
                                self.queue_review(cue,pcm,audio_start=start/RATE)
                    cursor, current_id = max(start, end-int(.2*RATE)), None
                    self.changed()
                    continue
                self.emit('quality_status','')
                with self.store.lock:
                    bounds = (min(x[0] for x in ranges), max(x[1] for x in ranges)) if ranges else (0, length)
                    if self.speakers:
                        # VAD can miss game voices. Attribute the entire ASR input,
                        # never just the small subset that VAD happened to detect.
                        bounds = (0,length)
                    if current_id is None:
                        current_id = max(self.store.cues, default=0) + 1
                        self.store.cues[current_id] = Cue(current_id, start/RATE+bounds[0], start/RATE+bounds[1], text)
                        last_translation_end = start
                        last_local_review_end = start
                    cue = self.store.cues[current_id]
                    if ranges == [] and '检测器未检出人声，已保留声音尝试转写' not in cue.reasons:
                        cue.reasons.append('检测器未检出人声，已保留声音尝试转写')
                    if cue.original != text:
                        previous = cue.original
                        if len(previous) >= 8 and SequenceMatcher(None, previous, text[:len(previous)]).ratio() < .65:
                            if '原文多次更新时差异较大' not in cue.reasons:
                                cue.reasons.append('原文多次更新时差异较大')
                        cue.original = text
                        cue.revision += 1
                        cue.local_review_status = ''
                        cue.local_review_original = ''
                        cue.local_review_translation = ''
                    cue.detected_language = response.get('detected_language', '')
                    cue.detected_language_source = response.get('detected_language_source', '')
                    corroborated = (content_key(cue.local_review_original) == content_key(text)
                                    and cue.local_review_status == '本地两模型转写一致，仍可能有误')
                    cue.uncertain = weak_transcript(text,ranges,length) and not corroborated
                    if cue.uncertain and '未检出人声且仅得到短词或重复片段，原文待复核' not in cue.reasons:
                        cue.reasons.append('未检出人声且仅得到短词或重复片段，原文待复核')
                    cue.end, cue.final, cue.asr_seconds = start/RATE+bounds[1], final, round(seconds, 3)
                    if self.speakers:
                        cue.speaker, cue.speaker_status = self.speaker_timeline.describe(cue.start, cue.end)
                    if final or end-last_translation_end >= 6*RATE:
                        self.queue_translation(cue)
                        last_translation_end = end
                    needs_review = cue.uncertain or ranges == [] or (final and bool(cue.reasons))
                    if needs_review and (final or end-last_local_review_end >= 6*RATE):
                        self.queue_local_review(cue,pcm)
                        last_local_review_end = end
                    if final:
                        self.archive_uncertain(cue,pcm,start/RATE)
                        self.queue_review(cue, pcm, audio_start=start/RATE)
                        cursor, current_id = end, None
                self.changed()
                self.emit('metrics', f'原文处理 {seconds:.2f}秒 · 待译 {len(self.translations.items)}段')
        finally:
            if self.stop.is_set() and current_id is not None:
                # Preserve the pending difficult span when the user stops early.
                # Do not start new inference during cancellation.
                start, end, pending = self.audio.read(cursor,min(self.audio.end,cursor+maximum))
                with self.store.lock:
                    cue = self.store.cues[current_id]
                    if pending and cue.reasons:
                        self.archive_uncertain(cue,pending,start/RATE)
            self.local_reviews.close(discard=self.stop.is_set())
            # A final corroborating review may still enqueue a translation.
            if self.local_review_thread:
                while self.local_review_thread.is_alive() and not self.stop.is_set():
                    self.local_review_thread.join(timeout=.2)
            self.translations.close(discard=self.stop.is_set())
            self.reviews.close(discard=self.stop.is_set())
            # At EOF finish accepted jobs. User Stop instead cancels the model processes.
            for thread in self.threads:
                while thread.is_alive() and not self.stop.is_set():
                    thread.join(timeout=.2)
            self.export()

    def export(self):
        cues = self.store.snapshot()
        (self.run_dir / 'captions.json').write_text(json.dumps(cues, ensure_ascii=False, indent=2), encoding='utf-8')
        blocks = []
        for cue in cues:
            if not cue['original']:
                continue
            text = cue['original']
            if self.speakers:
                text = '[' + speaker_name(cue['speaker']) + '] ' + text
            if cue['translated_revision'] == cue['revision'] and not cue['uncertain']:
                text += '\n' + cue['translation']
            if cue['uncertain']:
                text = '[待复核] ' + text
            blocks.append(f"{len(blocks)+1}\n{stamp(cue['start'])} --> {stamp(cue['end'])}\n{text}\n")
        (self.run_dir / 'bilingual.srt').write_text('\n'.join(blocks), encoding='utf-8-sig')


def stamp(seconds):
    value = max(0, round(seconds * 1000))
    hours, value = divmod(value, 3600000)
    minutes, value = divmod(value, 60000)
    seconds, ms = divmod(value, 1000)
    return f'{hours:02}:{minutes:02}:{seconds:02},{ms:03}'
