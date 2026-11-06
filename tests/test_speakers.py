import json
from pathlib import Path
import tempfile
import threading
import unittest

from engine import Pipeline, Cue, RATE
from speakers import VoiceRegistry, SpeakerTimeline
from test_pipeline import FakeModels, CONFIG


class SpeakerTests(unittest.TestCase):
    def test_returning_voice_retains_identity_after_other_voice(self):
        registry = VoiceRegistry()
        self.assertEqual(registry.assign([1,0,0],2)[0],0)
        self.assertEqual(registry.assign([0,1,0],2)[0],1)
        self.assertEqual(registry.assign([.95,.05,0],2)[0],0)
        self.assertEqual(len(registry.profiles),2)

    def test_ambiguous_or_short_voice_is_unknown_without_profile_pollution(self):
        registry = VoiceRegistry()
        registry.assign([1,0],2)
        registry.assign([0,1],2)
        before = [v.copy() for v in registry.profiles]
        self.assertIsNone(registry.assign([.7,.7],2)[0])
        self.assertIsNone(registry.assign([1,0],.3)[0])
        for a,b in zip(before,registry.profiles):
            self.assertTrue((a==b).all())

    def test_profile_cap_does_not_force_unrelated_voice_into_existing_person(self):
        registry = VoiceRegistry(capacity=1)
        registry.assign([1,0],2)
        self.assertIsNone(registry.assign([0,1],2)[0])

    def test_timeline_does_not_label_mixed_or_not_yet_analyzed_audio(self):
        timeline = SpeakerTimeline()
        timeline.update({'start':0,'end':10,'turns':[
            {'start':1,'end':5,'speaker':0},{'start':6,'end':9,'speaker':1}]})
        self.assertEqual(timeline.describe(1,5)[0],0)
        self.assertIsNone(timeline.describe(1,9)[0])
        self.assertIsNone(timeline.describe(6,12)[0])
        self.assertEqual(timeline.boundary(0,10),5.5)

    def test_boundary_preserves_unidentified_turn_before_confirmed_new_speaker(self):
        timeline = SpeakerTimeline()
        timeline.update({'start':0,'end':12,'turns':[
            {'start':3,'end':7,'speaker':0}, {'start':8,'end':9,'speaker':None},
            {'start':10,'end':12,'speaker':1}]})
        self.assertEqual(timeline.boundary(0,12),7.5)

    def test_rolling_window_keeps_prior_history_and_revises_recent_span(self):
        timeline = SpeakerTimeline()
        timeline.update({'start':0,'end':10,'turns':[{'start':1,'end':9,'speaker':0}]})
        timeline.update({'start':5,'end':20,'turns':[{'start':5,'end':12,'speaker':1}]})
        self.assertEqual(timeline.describe(1,4)[0],0)
        self.assertEqual(timeline.describe(6,10)[0],1)
        self.assertTrue(all(t['end']>t['start'] for t in timeline.turns))

    def test_failed_speaker_analysis_does_not_stop_asr_or_translation(self):
        class BrokenSpeaker:
            def analyze(self,*args):
                raise RuntimeError('test worker failure')
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as directory:
            pipe = Pipeline(FakeModels(),CONFIG,directory,lambda *a:None,threading.Event(),speakers=BrokenSpeaker())
            pipe.audio.append(bytes(3*32000))
            pipe.audio.finish()
            pipe.run()
            self.assertTrue(pipe.store.snapshot()[0]['translation'])
            self.assertIn('speaker_error',(Path(directory)/'events.jsonl').read_text())

    def test_known_boundary_splits_real_asr_input_and_exports_matching_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {**CONFIG,'max_segment_seconds':12}
            model = FakeModels()
            pipe = Pipeline(model,config,directory,lambda *a:None,threading.Event())
            pipe.speakers = object()
            pipe.speaker_timeline.update({'start':0,'end':12,'turns':[
                {'start':0,'end':5,'speaker':0},{'start':6,'end':12,'speaker':1}]})
            # Drive the normal ASR loop using a fixed, already analyzed timeline.
            pipe.speaker_worker = lambda:None
            pipe.audio.append(bytes(12*32000))
            pipe.audio.finish()
            pipe.run()
            self.assertEqual([c[1] for c in model.calls if c[0]=='asr'],[5.5,6.5])
            cues = pipe.store.snapshot()
            self.assertEqual([c['speaker'] for c in cues],[0,1])
            srt = (Path(directory)/'bilingual.srt').read_text(encoding='utf-8-sig')
            self.assertIn('[A]',srt)
            self.assertIn('[B]',srt)

    def test_change_discovered_during_asr_revises_before_finalizing(self):
        with tempfile.TemporaryDirectory() as directory:
            class DelayedBoundary(FakeModels):
                def transcribe(self,*args,**kwargs):
                    result = super().transcribe(*args,**kwargs)
                    if len(self.calls)==1:
                        pipe.speaker_timeline.update({'start':0,'end':12,'turns':[
                            {'start':0,'end':5,'speaker':0},{'start':6,'end':12,'speaker':1}]})
                    return result
            model = DelayedBoundary()
            pipe = Pipeline(model,{**CONFIG,'max_segment_seconds':12},directory,lambda *a:None,threading.Event())
            pipe.speakers = object()
            pipe.speaker_worker = lambda:None
            pipe.audio.append(bytes(12*32000))
            pipe.audio.finish()
            pipe.run()
            self.assertEqual([c[1] for c in model.calls if c[0]=='asr'],[12,5.5,6.5])
            self.assertEqual([c['speaker'] for c in pipe.store.snapshot()],[0,1])

    def test_vad_missing_first_voice_cannot_assign_whole_transcript_to_last_voice(self):
        class BadVad(FakeModels):
            def transcribe(self,*args,**kwargs):
                return {'text':'Both speakers are in this text.', 'vad_segments':{'crispasr_vad':{
                    'sample_rate':RATE,'slices':[{'start':9*RATE,'end':10*RATE}]}}}
        with tempfile.TemporaryDirectory() as directory:
            pipe = Pipeline(BadVad(),{**CONFIG,'max_segment_seconds':12},directory,lambda *a:None,threading.Event())
            pipe.speakers = object()
            pipe.speaker_worker = lambda:None
            pipe.speaker_timeline.update({'start':0,'end':12,'turns':[
                {'start':1,'end':5,'speaker':None},{'start':9,'end':10,'speaker':1}]})
            pipe.audio.append(bytes(12*32000))
            pipe.audio.finish()
            pipe.run()
            cue = pipe.store.snapshot()[0]
            self.assertEqual((cue['start'],cue['end']),(0,12))
            self.assertIsNone(cue['speaker'])


if __name__=='__main__':
    unittest.main(verbosity=2)
