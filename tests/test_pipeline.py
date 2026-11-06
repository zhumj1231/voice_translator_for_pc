import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from backends import CloudReviewer
from engine import AudioBuffer, CaptionStore, Cue, LatestJobs, Pipeline, RATE, vad_bounds


def response(text, seconds):
    return {'text': text, 'vad_segments': {'crispasr_vad': {'sample_rate': RATE,
            'slices': [{'start': 0, 'end': int(seconds*RATE)}] if text else []}}}


class FakeModels:
    def __init__(self, text='今日は晴れです。'):
        self.text = text
        self.calls = []

    def transcribe(self, pcm, language, vad=True):
        self.calls.append(('asr', len(pcm)/32000))
        return response(self.text, len(pcm)/32000)

    def translate(self, text, context):
        self.calls.append(('translate', text, context))
        return '今天是晴天。'


class FakeCloud:
    available = True

    def __init__(self):
        self.calls = []

    def review(self, pcm, language):
        self.calls.append(len(pcm)/32000)
        return '明日は雨です。'


CONFIG = {'language': 'ja', 'refresh_seconds': 3, 'max_segment_seconds': 6, 'vad': True,
          'cloud': {'enabled': False, 'provider': 'fake', 'mode': 'all', 'max_audio_seconds_per_session': 30}}


class PipelineTests(unittest.TestCase):
    def test_late_translation_cannot_replace_revised_source(self):
        store = CaptionStore()
        store.cues[1] = Cue(1, 0, 6, '修正后的原文', revision=2)
        self.assertFalse(store.translated(1, 1, '对应旧原文的译文'))
        self.assertEqual(store.cues[1].translation, '')
        self.assertTrue(store.translated(1, 1, '对应旧原文的译文', '旧版原文'))
        self.assertEqual(store.cues[1].original, '修正后的原文')
        self.assertEqual(store.cues[1].translated_from, '旧版原文')
        self.assertEqual(store.cues[1].translation_status, '此前原文译文')
        self.assertTrue(store.translated(1, 2, '对应新版的译文'))
        self.assertFalse(store.translated(1, 1, '更晚返回的旧译文', '旧版原文'))
        self.assertEqual(store.cues[1].translation, '对应新版的译文')

    def test_slow_translation_is_shown_with_its_own_source_before_final_asr(self):
        store = CaptionStore()
        store.cues[1] = Cue(1, 0, 6, '第一句。', revision=1)
        in_flight = {'revision': 1, 'source': '第一句。'}
        # ASR expands while translation is running.
        store.cues[1].original = '第一句。第二句。'
        store.cues[1].revision = 2
        self.assertTrue(store.translated(1, in_flight['revision'], 'First sentence.', in_flight['source']))
        self.assertEqual(store.cues[1].translated_from, '第一句。')
        self.assertNotEqual(store.cues[1].translated_revision, store.cues[1].revision)

    def test_queue_coalesces_revisions_and_bounds_backlog(self):
        jobs = LatestJobs(2)
        jobs.put({'id': 1, 'revision': 1})
        removed = jobs.put({'id': 1, 'revision': 2})
        self.assertEqual(removed, [{'id': 1, 'revision': 1}])
        jobs.put({'id': 2, 'revision': 1})
        removed = jobs.put({'id': 3, 'revision': 1})
        self.assertEqual(removed, [{'id': 1, 'revision': 2}])
        self.assertEqual(jobs.get()['id'], 2)
        self.assertEqual(jobs.get()['id'], 3)
        jobs.close()
        self.assertIsNone(jobs.get())

    def test_context_excludes_future_drafts_and_stale_translation(self):
        store = CaptionStore()
        store.cues[1] = Cue(1, 0, 3, '前文', final=True, revision=2,
                           translation='过期译文', translated_revision=1)
        store.cues[2] = Cue(2, 3, 6, '待译句', final=True)
        store.cues[3] = Cue(3, 6, 9, '未来内容', final=True)
        self.assertEqual(store.context(2), [{'原文': '前文', '译文': ''}])

    def test_audio_ring_retains_absolute_sample_positions(self):
        audio = AudioBuffer(seconds=1)
        audio.append(bytes(RATE*2))
        audio.append(b'\x01\x00' * RATE)
        start, end, pcm = audio.read(0, RATE*2)
        self.assertEqual((start, end, len(pcm)), (RATE, RATE*2, RATE*2))
        self.assertEqual(pcm[:2], b'\x01\x00')

    def test_silence_produces_no_caption_or_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModels(text='')
            pipe = Pipeline(model, CONFIG, directory, lambda *args: None, threading.Event())
            pipe.audio.append(bytes(3*32000))
            pipe.audio.finish()
            pipe.run()
            self.assertEqual(pipe.store.snapshot(), [])
            self.assertEqual(len(model.calls), 1)

    def test_slow_asr_skips_old_audio_with_explicit_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModels()
            pipe = Pipeline(model, CONFIG, directory, lambda *args: None, threading.Event())
            pipe.audio.append(bytes(30*32000))
            pipe.audio.finish()
            pipe.run()
            self.assertEqual(model.calls[0], ('asr', 6.0))
            events = [json.loads(line) for line in (Path(directory)/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(events[0]['event'], 'audio_gap')
            self.assertEqual((events[0]['start'], events[0]['end']), (0, 24))

    def test_cloud_difference_does_not_silently_rewrite_caption(self):
        with tempfile.TemporaryDirectory() as directory:
            cloud = FakeCloud()
            pipe = Pipeline(FakeModels(), CONFIG, directory, lambda *args: None, threading.Event(), cloud)
            pipe.audio.append(bytes(3*32000))
            pipe.audio.finish()
            pipe.run()
            cue = pipe.store.snapshot()[0]
            self.assertEqual(cue['original'], '今日は晴れです。')
            self.assertEqual(cue['review_original'], '明日は雨です。')
            self.assertEqual(cloud.calls, [3.0])
            self.assertTrue((Path(directory)/'bilingual.srt').is_file())

    def test_unconfigured_cloud_makes_no_request(self):
        cloud = CloudReviewer({'enabled': False})
        self.assertFalse(cloud.available)
        with tempfile.TemporaryDirectory() as directory:
            pipe = Pipeline(FakeModels(), CONFIG, directory, lambda *args: None, threading.Event(), cloud)
            pipe.audio.append(bytes(3*32000))
            pipe.audio.finish()
            pipe.run()
            self.assertEqual(pipe.review_seconds, 0)

    def test_cloud_audio_limit_is_enforced_before_sending(self):
        with tempfile.TemporaryDirectory() as directory:
            cloud = FakeCloud()
            config = {**CONFIG, 'cloud': {**CONFIG['cloud'], 'max_audio_seconds_per_session': 2}}
            pipe = Pipeline(FakeModels(), config, directory, lambda *args: None, threading.Event(), cloud)
            pipe.audio.append(bytes(3*32000))
            pipe.audio.finish()
            pipe.run()
            self.assertEqual(cloud.calls, [])
            self.assertIn('上限', pipe.store.snapshot()[0]['review_status'])

    def test_vad_schema_and_missing_metadata(self):
        # Synthetic schema examples; no captured audio or private benchmark files.
        silent = {'vad_segments': {'crispasr_vad': {'sample_rate': 16000, 'slices': []}}}
        speech = {'vad_segments': {'crispasr_vad': {'sample_rate': 16000,
                  'slices': [{'start': 0, 'end': 192000}]}}}
        self.assertEqual(vad_bounds(silent), [])
        self.assertEqual(vad_bounds(speech), [(0, 12)])
        self.assertIsNone(vad_bounds({'text': ''}))

    def test_missed_voice_is_not_discarded_just_because_vad_is_empty(self):
        class MissedVoice(FakeModels):
            def transcribe(self, *args, **kwargs):
                return {'text': '小声でも話しています。', 'vad_segments': {
                    'crispasr_vad': {'sample_rate': RATE, 'slices': []}},
                    'audio_gate': {'silent': False, 'tail_quiet': False}}
        with tempfile.TemporaryDirectory() as directory:
            pipe = Pipeline(MissedVoice(), CONFIG, directory, lambda *args: None, threading.Event())
            pipe.audio.append(bytes(3*32000))
            pipe.audio.finish()
            pipe.run()
            self.assertEqual(pipe.store.snapshot()[0]['original'], '小声でも話しています。')
            self.assertTrue(pipe.store.snapshot()[0]['reasons'])

    def test_a_second_session_is_blocked_until_the_first_releases_runtime(self):
        import app
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(app, 'DATA', Path(directory)):
            first, second = app.Session(CONFIG, lambda *a: None), app.Session(CONFIG, lambda *a: None)
            try:
                first.acquire_runtime()
                with self.assertRaisesRegex(RuntimeError, '已有一个字幕服务'):
                    second.acquire_runtime()
                first.release_runtime()
                second.acquire_runtime()
            finally:
                first.release_runtime()
                second.release_runtime()

    def test_dedicated_cloud_asr_uses_verbatim_audio_without_reference_text(self):
        cfg = {'enabled': True, 'provider': 'gemini_generate_content',
               'endpoint': 'https://generativelanguage.googleapis.com/v1beta',
               'model': 'gemini-3.5-transcribe', 'key_env': 'SUBTITLE_TEST_KEY'}
        result = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': '音声'}]}}]}
        with mock.patch.dict('os.environ', {'SUBTITLE_TEST_KEY': 'test-only'}), mock.patch('backends.gemini_post', return_value=result) as post:
            self.assertEqual(CloudReviewer(cfg).review(bytes(32000), 'ja'), '音声')
            payload = post.call_args.args[1]
            self.assertEqual(payload['generationConfig']['audioTranscriptionConfig']['mode'], 'VERBATIM')
            self.assertEqual(len(payload['contents'][0]['parts']), 1)
            self.assertIn('inlineData', payload['contents'][0]['parts'][0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
