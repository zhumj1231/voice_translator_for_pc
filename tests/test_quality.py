"""Behavior checks for missed game speech, review races and diagnostic audio."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
import wave

from audio_quality import weak_transcript
from engine import Pipeline, Cue, CaptionStore, RATE
from test_pipeline import CONFIG, FakeModels

LOUD = b'\xe8\x03' * RATE


class WeakModel(FakeModels):
    def transcribe(self, pcm, language, vad=True):
        self.calls.append(('asr', len(pcm)/32000))
        return {'text': self.text, 'vad_segments': {'crispasr_vad': {
            'sample_rate': RATE, 'slices': []}}, 'audio_gate': {'silent': False}}


class Reviewer:
    def __init__(self, text='.', event='<|BGM|>', language='<|ja|>'):
        self.result = {'text': text, 'audio_event': event, 'language': language}
        self.closed = False

    def analyze(self, pcm, start):
        return self.result

    def close(self):
        self.closed = True


class QualityTests(unittest.TestCase):
    def run_clip(self, model, reviewer, directory, **config):
        pipe = Pipeline(model, {**CONFIG, **config}, directory, lambda *a: None,
                        threading.Event(), local_reviewer=reviewer)
        pipe.audio.append(LOUD*3)
        pipe.audio.finish()
        pipe.run()
        return pipe

    def test_no_vad_short_text_is_kept_but_not_translated_or_used_as_context(self):
        with tempfile.TemporaryDirectory() as directory:
            model = WeakModel('中国。')
            reviewer = Reviewer()
            pipe = self.run_clip(model, reviewer, directory)
            cue = pipe.store.snapshot()[0]
            self.assertEqual(cue['original'], '中国。')
            self.assertTrue(cue['uncertain'])
            self.assertEqual(cue['translation'], '')
            self.assertEqual(cue['local_review_original'], '')
            self.assertIn('语音漏检', cue['local_review_status'])
            self.assertEqual(pipe.store.context(2), [])
            self.assertFalse(any(c[0]=='translate' for c in model.calls))
            self.assertIn('[待复核]', (Path(directory)/'bilingual.srt').read_text(encoding='utf-8-sig'))
            self.assertTrue(reviewer.closed)

    def test_final_matching_review_still_gets_translation_after_asr_eof(self):
        with tempfile.TemporaryDirectory() as directory:
            pipe = self.run_clip(WeakModel('こんにちは。'),
                                 Reviewer('こんにちは！', '<|Speech|>'), directory)
            cue = pipe.store.snapshot()[0]
            self.assertFalse(cue['uncertain'])
            self.assertTrue(cue['translation'])
            self.assertEqual(cue['translated_revision'], cue['revision'])

    def test_different_review_is_a_separate_candidate_with_paired_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            pipe = self.run_clip(WeakModel('中国。'),
                                 Reviewer('こちらです。', '<|Speech|>'), directory)
            cue = pipe.store.snapshot()[0]
            self.assertEqual(cue['original'], '中国。')
            self.assertTrue(cue['uncertain'])
            self.assertEqual(cue['translation'], '')
            self.assertEqual(cue['local_review_original'], 'こちらです。')
            self.assertTrue(cue['local_review_translation'])

    def test_legacy_setting_does_not_reject_recognized_other_language(self):
        with tempfile.TemporaryDirectory() as directory:
            pipe = self.run_clip(WeakModel('中国。'),
                                 Reviewer('中国。', '<|Speech|>', '<|zh|>'), directory)
            self.assertFalse(pipe.store.snapshot()[0]['uncertain'])
            self.assertTrue(pipe.store.snapshot()[0]['translation'])

    def test_auto_accepts_recognized_speech_but_not_unknown_language(self):
        for language, expected_uncertain in [('<|ja|>', False), ('<|en|>', False), ('', True)]:
            with self.subTest(language=language), tempfile.TemporaryDirectory() as directory:
                pipe = self.run_clip(WeakModel('hello'), Reviewer('hello', '<|Speech|>', language),
                                     directory, language='auto')
                self.assertEqual(pipe.store.snapshot()[0]['uncertain'], expected_uncertain)

    def test_bgm_event_cannot_discard_substantial_target_language_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            pipe = self.run_clip(WeakModel('中国。'),
                Reviewer('こちらには話している人がいます。','<|BGM|>'),directory)
            cue = pipe.store.snapshot()[0]
            self.assertEqual(cue['local_review_original'],'こちらには話している人がいます。')
            self.assertIn('背景声',cue['local_review_status'])
            self.assertTrue(cue['local_review_translation'])
            self.assertTrue(cue['uncertain'])
            self.assertEqual(cue['original'],'中国。')

    def test_stale_review_cannot_change_newer_source(self):
        with tempfile.TemporaryDirectory() as directory:
            model = WeakModel('旧文')
            pipe = Pipeline(model, CONFIG, directory, lambda *a: None, threading.Event())
            cue = pipe.store.cues[1] = Cue(1,0,3,'旧文',uncertain=True)
            class RevisingReviewer(Reviewer):
                def analyze(self, *args):
                    cue.original, cue.revision = '新版原文', 2
                    return super().analyze(*args)
            pipe.local_reviewer = RevisingReviewer('旧文','<|Speech|>')
            pipe.queue_local_review(cue, LOUD*3)
            pipe.local_reviews.close()
            pipe.local_review_worker()
            self.assertEqual(cue.local_review_original, '')
            self.assertEqual(cue.original, '新版原文')
            self.assertTrue(cue.uncertain)
            self.assertEqual(model.calls, [])

    def test_large_final_revision_gets_review_even_when_vad_detected_speech(self):
        with tempfile.TemporaryDirectory() as directory:
            class UnstableModel(FakeModels):
                def transcribe(self, *args, **kwargs):
                    result = super().transcribe(*args, **kwargs)
                    if len(self.calls)==1:
                        pipe.audio.append(LOUD*3)
                        pipe.audio.finish()
                    else:
                        result['text'] = 'まったく異なる文章になりました。'
                    return result
            model = UnstableModel('こちらは最初の長い文章です。')
            pipe = Pipeline(model,CONFIG,directory,lambda *a:None,threading.Event(),
                            local_reviewer=Reviewer())
            pipe.audio.append(LOUD*3)
            pipe.run()
            self.assertTrue(pipe.store.snapshot()[0]['local_review_status'])

    def test_non_silent_empty_draft_keeps_pcm_for_next_asr_call(self):
        with tempfile.TemporaryDirectory() as directory:
            class RecoveringModel(WeakModel):
                def transcribe(self, pcm, *args, **kwargs):
                    result = super().transcribe(pcm, *args, **kwargs)
                    if len(self.calls)==1:
                        result['text'] = ''
                        pipe.audio.append(LOUD*3)
                        pipe.audio.finish()
                    return result
            model = RecoveringModel('ここには声があります。')
            pipe = Pipeline(model,CONFIG,directory,lambda *a:None,threading.Event())
            pipe.audio.append(LOUD*3)
            pipe.run()
            self.assertEqual([c[1] for c in model.calls if c[0]=='asr'], [3,6])
            self.assertEqual(pipe.store.snapshot()[0]['start'], 0)
            self.assertTrue(pipe.store.snapshot()[0]['translation'])
            self.assertIn('asr_empty_retained',(Path(directory)/'events.jsonl').read_text())

    def test_empty_final_is_visible_and_archives_original_pcm_with_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            pipe = self.run_clip(WeakModel(''), Reviewer(), directory,
                                 diagnostics={'save_uncertain_audio':True,'max_audio_seconds':3})
            cue = pipe.store.snapshot()[0]
            self.assertEqual(cue['original'], '')
            self.assertTrue(cue['uncertain'])
            path = Path(cue['audio_clip'])
            with wave.open(str(path),'rb') as audio:
                self.assertEqual(audio.readframes(3*RATE), LOUD*3)
            meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            self.assertEqual((meta['audio_start'],meta['audio_end']), (0,3))
            pipe.archive_uncertain(pipe.store.cues[1], LOUD*3, 0)
            self.assertEqual(pipe.saved_audio_seconds,3)
            pipe.archive_uncertain(Cue(2,3,6,'?',reasons=['uncertain']), LOUD*3,3)
            self.assertEqual(len(list(path.parent.glob('*.wav'))),1)

    def test_empty_final_after_draft_marks_whole_span_uncertain(self):
        with tempfile.TemporaryDirectory() as directory:
            class VanishingModel(WeakModel):
                def transcribe(self, pcm,*args,**kwargs):
                    result = super().transcribe(pcm,*args,**kwargs)
                    if len(self.calls)==1:
                        pipe.audio.append(LOUD*3)
                        pipe.audio.finish()
                    else:
                        result['text'] = ''
                    return result
            model = VanishingModel('ここには声があります。')
            pipe = Pipeline(model,CONFIG,directory,lambda *a:None,threading.Event())
            pipe.audio.append(LOUD*3)
            pipe.run()
            cue = pipe.store.snapshot()[0]
            self.assertEqual(cue['end'],6)
            self.assertTrue(cue['uncertain'])
            self.assertEqual(cue['translation'],'')

    def test_pending_translation_cannot_publish_newly_uncertain_source(self):
        store = CaptionStore()
        store.cues[1] = Cue(1,0,3,'待定文字',uncertain=True)
        self.assertFalse(store.translated(1,1,'不应显示','待定文字'))

    def test_stop_preserves_pending_difficult_audio_without_new_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = threading.Event()
            class StoppedModel(WeakModel):
                def transcribe(self,*args,**kwargs):
                    result = super().transcribe(*args,**kwargs)
                    if len(self.calls)==1:
                        pipe.audio.append(LOUD*3)
                    else:
                        stop.set()
                    return result
            model = StoppedModel('あああ')
            pipe = Pipeline(model,{**CONFIG,'max_segment_seconds':20,
                'diagnostics':{'save_uncertain_audio':True}},directory,lambda *a:None,stop)
            pipe.audio.append(LOUD*3)
            pipe.run()
            cue = pipe.store.snapshot()[0]
            self.assertFalse(cue['final'])
            self.assertTrue(Path(cue['audio_clip']).is_file())
            self.assertEqual(pipe.saved_audio_seconds,6)
            self.assertEqual(len(model.calls),2)

    def test_detected_short_speech_and_missing_vad_are_not_automatically_rejected(self):
        self.assertFalse(weak_transcript('はい。',[(0,2)],3))
        self.assertFalse(weak_transcript('Yes.',None,3))
        self.assertTrue(weak_transcript('はい。',[],3))


if __name__ == '__main__':
    unittest.main(verbosity=2)
