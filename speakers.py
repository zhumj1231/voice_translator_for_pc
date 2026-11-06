"""Anonymous, session-scoped speaker tracking. No names or persistent voice profiles."""
from collections import Counter
from dataclasses import dataclass
import base64
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

import numpy as np

from paths import ROOT, SPEAKER_MODELS
RATE = 16000


def normalized(vector):
    v = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    return v/norm if norm > 1e-8 and np.isfinite(v).all() else None


class VoiceRegistry:
    """Conservative cosine matching; ambiguous snippets never update a profile."""
    def __init__(self, match=.4, margin=.08, capacity=12):
        self.match, self.margin, self.capacity = match, margin, capacity
        self.profiles = []
        self.durations = []

    def improve(self, index, vector, duration):
        v = normalized(vector)
        if (v is not None and duration > self.durations[index]+.5
                and float(np.dot(v,self.profiles[index])) >= self.match):
            self.profiles[index] = normalized(.35*self.profiles[index]+.65*v)
            self.durations[index] = duration

    def assign(self, vector, duration, update=True):
        v = normalized(vector)
        if v is None or duration < .85:
            return None, '声音太短或特征不足'
        scores = sorted(((float(np.dot(v, p)), i) for i, p in enumerate(self.profiles)), reverse=True)
        if scores and scores[0][0] >= self.match:
            score, index = scores[0]
            if len(scores) > 1 and score-scores[1][0] < self.margin:
                return None, '与多个声音相似'
            if update and duration >= 1.5 and score >= self.match+.1:
                self.improve(index,v,duration)
            return index, '声音匹配'
        if scores and scores[0][0] >= self.match-.1:
            return None, '声音匹配待确认'
        if not update or duration < 1.2 or len(self.profiles) >= self.capacity:
            return None, '声音样本不足或角色数量达到上限'
        self.profiles.append(v)
        self.durations.append(duration)
        return len(self.profiles)-1, '新声音'


def name(speaker):
    return '待确认' if speaker is None else chr(65+speaker)


def overlap(a, b, c, d):
    return max(0., min(b, d)-max(a, c))


class SpeakerTimeline:
    def __init__(self):
        self.lock = threading.RLock()
        self.turns = []
        self.analyzed_end = 0.
        self.status = '角色待确认'

    def update(self, result):
        with self.lock:
            # Only replace the analyzed range, and keep recent history bounded.
            start, end = result['start'], result['end']
            previous = [{**t, 'end': min(t['end'], start)} for t in self.turns
                        if t['start'] < start and t['end'] > end-90]
            self.turns = previous + result['turns']
            self.analyzed_end = end
            self.status = '已分析'

    def describe(self, start, end):
        with self.lock:
            if self.status == '角色分析失败':
                return None, self.status
            weights = Counter()
            unknown = 0.
            for t in self.turns:
                seconds = overlap(start, end, t['start'], t['end'])
                if t['speaker'] is None:
                    unknown += seconds
                else:
                    weights[t['speaker']] += seconds
            total = sum(weights.values())+unknown
            if end > self.analyzed_end+.25:
                return None, '角色分析中'
            if not weights or total < .5:
                return None, '角色待确认'
            speaker, seconds = weights.most_common(1)[0]
            # A known change must not be hidden by a long dominant speaker.
            others = sum(v for k,v in weights.items() if k != speaker)
            if others >= .5:
                return None, '多位说话人，尚未逐句分开'
            if seconds/total < .8:
                return None, '角色待确认'
            return speaker, '声音分组，可能有误'

    def boundary(self, start, end):
        with self.lock:
            if self.status == '角色分析失败':
                return None
            known = [t for t in self.turns if t['speaker'] is not None
                     and overlap(start, end, t['start'], t['end']) >= .5]
            if not known:
                return None
            first = known[0]
            for t in known[1:]:
                if t['speaker'] != first['speaker'] and t['start'] >= start+.8:
                    # Leave a little right context to confirm the change.
                    if min(end, self.analyzed_end)-t['start'] >= .85:
                        # An unlabelled turn between the two voices may already
                        # belong to the next speaker. Never cut through its middle.
                        next_start = min((u['start'] for u in self.turns
                                          if u['start'] >= first['end'] and u['start'] <= t['start']),
                                         default=t['start'])
                        return max(start+.8, (min(first['end'], next_start)+next_start)/2)
                first = t
            return None


class SpeakerClient:
    """A separate process is required: sherpa's process binding holds Python's GIL."""
    def __init__(self, config, run_dir, stop, service='speaker_service.py', log_name='speakers'):
        self.config, self.run_dir, self.stop = config, Path(run_dir), stop
        self.process = None
        self.log = None
        self.responses = queue.Queue()
        self.lifecycle = threading.Lock()
        self.reader = None
        self.service, self.log_name = service, log_name

    def receive(self, timeout):
        deadline = time.monotonic()+timeout
        while not self.stop.is_set():
            try:
                result = self.responses.get(timeout=min(.1, max(.01, deadline-time.monotonic())))
                if result is None:
                    raise RuntimeError('后台音频进程退出，见 '+self.log_name+'.log')
                if 'error' in result:
                    raise RuntimeError(result['error'])
                return result
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise TimeoutError('后台音频分析超时')
        raise InterruptedError('已停止')

    def start(self):
        with self.lifecycle:
            if self.stop.is_set():
                raise InterruptedError('已停止')
            self.log = (self.run_dir/(self.log_name+'.log')).open('wb')
            self.process = subprocess.Popen([sys.executable, '-X', 'utf8', str(ROOT/self.service)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                creationflags=subprocess.CREATE_NO_WINDOW)
        def read():
            try:
                for line in self.process.stdout:
                    self.responses.put(json.loads(line))
            except (OSError, ValueError):
                pass
            finally:
                self.responses.put(None)
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        return self.receive(40)

    def analyze(self, pcm, start, closed=False):
        request = {'start': start, 'pcm': base64.b64encode(pcm).decode('ascii'),
                   'closed': closed, 'match': self.config.get('match_threshold', .4)}
        self.process.stdin.write((json.dumps(request)+'\n').encode())
        self.process.stdin.flush()
        return self.receive(25)

    def close(self):
        with self.lifecycle:
            if self.process:
                if self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=3)
                for handle in (self.process.stdin, self.process.stdout):
                    if handle:
                        handle.close()
            if self.log:
                self.log.close()


class SpeakerAnalyzer:
    def __init__(self):
        import sherpa_onnx as so
        model = str(SPEAKER_MODELS/'3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx')
        emb_config = so.SpeakerEmbeddingExtractorConfig(model=model, num_threads=2, provider='cpu')
        config = so.OfflineSpeakerDiarizationConfig(
            segmentation=so.OfflineSpeakerSegmentationModelConfig(
                pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(SPEAKER_MODELS/'sherpa-onnx-pyannote-segmentation-3-0/model.onnx')),
                num_threads=2, provider='cpu'),
            embedding=emb_config, clustering=so.FastClusteringConfig(num_clusters=-1, threshold=.5),
            min_duration_on=.3, min_duration_off=.3)
        if not config.validate():
            raise RuntimeError('缺少角色分析模型')
        self.model = so.OfflineSpeakerDiarization(config)
        self.extractor = so.SpeakerEmbeddingExtractor(emb_config)
        self.registry = VoiceRegistry()
        self.history = []

    def analyze(self, pcm, start, closed=False):
        samples = np.frombuffer(pcm, '<i2').astype(np.float32)/32768
        duration, turns = len(samples)/RATE, []
        if not len(samples) or np.max(np.abs(samples)) < .001:
            return {'start': start, 'end': start+duration, 'turns': []}
        raw = self.model.process(samples).sort_by_start_time()
        # For sub-10-second inputs the backend exposes overlapping local tracks.
        # Union overlapping detections, then compute independent voice features;
        # do not interpret local track IDs as stable people.
        groups = []
        for t in raw:
            a, b = max(0., t.start), min(duration, t.end)
            if b <= a:
                continue
            if groups and (a < groups[-1][1] or
                           (t.speaker == groups[-1][2] and a <= groups[-1][1]+.18)):
                groups[-1][1] = max(groups[-1][1], b)
            else:
                groups.append([a, b, t.speaker])
        anchored_groups = []
        for a,b,local_id in groups:
            # A long new detection may merge earlier distinct voices. Retain
            # their established boundary instead of contaminating both profiles.
            prior = [t for t in self.history if t['speaker'] is not None
                     and overlap(start+a,start+b,t['start'],t['end']) >= .85]
            cuts = []
            for left,right in zip(prior,prior[1:]):
                cut = (left['end']+right['start'])/2-start
                if left['speaker'] != right['speaker'] and a+.5 < cut < b-.5:
                    cuts.append(cut)
            positions = [a]+sorted(set(cuts))+[b]
            anchored_groups.extend((x,y,local_id) for x,y in zip(positions,positions[1:]))
        for a, b, _ in anchored_groups:
            abs_a, abs_b = start+a, start+b
            stream = self.extractor.create_stream()
            stream.accept_waveform(sample_rate=RATE, waveform=samples[round(a*RATE):round(b*RATE)])
            stream.input_finished()
            v = self.extractor.compute(stream) if self.extractor.is_ready(stream) else []
            old = max(self.history, key=lambda t: overlap(abs_a,abs_b,t['start'],t['end']), default=None)
            old_overlap = overlap(abs_a,abs_b,old['start'],old['end']) if old else 0.
            repeat = bool(old and old_overlap >= .7*(b-a))
            speaker, status = self.registry.assign(v, b-a, update=not repeat)
            if repeat and old['speaker'] is not None:
                # Exact repeated audio anchors IDs even if a window edge changes its feature.
                speaker, status = old['speaker'], '连续片段沿用编号'
                self.registry.improve(speaker,v,b-a)
            turns.append({'start': round(abs_a,3), 'end': round(abs_b,3),
                          'speaker': speaker, 'status': status})
        self.history = [t for t in self.history if t['end'] <= start and t['end'] >= start-60]+turns
        return {'start': start, 'end': start+duration, 'turns': turns,
                'speaker_count': len(self.registry.profiles)}
