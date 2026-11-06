"""Concurrency and spending guard regressions; no API requests."""
import tempfile
import threading
import unittest
from pathlib import Path
from engine import Pipeline, Cue, LatestJobs
from test_pipeline import FakeModels, FakeCloud, CONFIG


class RuntimeCloudTests(unittest.TestCase):
    def test_consecutive_failures_display_reason_and_stop_further_attempts(self):
        reason = 'Clash 当前为全局模式，请切换为规则模式；音频未发送'
        class BlockedCloud(FakeCloud):
            def review(self, *args):
                self.calls.append('attempt')
                raise RuntimeError(reason)
        with tempfile.TemporaryDirectory() as d:
            cloud, updates = BlockedCloud(), []
            pipe = Pipeline(FakeModels(), CONFIG, d, lambda k,v:updates.append((k,v)), threading.Event(), cloud)
            for cue_id in (1,2):
                pipe.reviews = LatestJobs(1)
                pipe.store.cues[cue_id] = Cue(cue_id,0,3,'原文')
                pipe.queue_review(pipe.store.cues[cue_id], bytes(32000))
                pipe.reviews.close()
                pipe.review_worker()
            self.assertEqual(len(cloud.calls), 2)
            self.assertIn(reason, pipe.store.cues[2].review_status)
            pipe.store.cues[3] = Cue(3,0,3,'原文')
            pipe.queue_review(pipe.store.cues[3], bytes(32000))
            self.assertIn(reason, pipe.store.cues[3].review_status)
            self.assertTrue(any(k=='cloud_status' and reason in v for k,v in updates))
            self.assertTrue(any(k=='cloud_status' and '含失败' in v for k,v in updates))

    def test_cloud_text_visible_before_translation_finishes(self):
        translating, release = threading.Event(), threading.Event()
        class Models(FakeModels):
            def translate(self, text, context):
                translating.set()
                release.wait(3)
                return '译文'
        snapshots = []
        with tempfile.TemporaryDirectory() as d:
            pipe = Pipeline(Models(), CONFIG, d, lambda k,v: snapshots.append(v) if k=='captions' else None,
                            threading.Event(), FakeCloud())
            pipe.store.cues[1] = Cue(1, 0, 3, '原文')
            pipe.queue_review(pipe.store.cues[1], bytes(3*32000))
            pipe.reviews.close()
            worker = threading.Thread(target=pipe.review_worker)
            worker.start()
            try:
                self.assertTrue(translating.wait(2))
                self.assertTrue(any(s[0]['review_original'] and not s[0]['review_translation'] for s in snapshots))
            finally:
                release.set()
                worker.join(3)
            self.assertEqual(pipe.store.cues[1].review_translation, '译文')

    def test_empty_is_not_an_alternative_transcript(self):
        class Empty(FakeCloud):
            def review(self, *args): return ''
        with tempfile.TemporaryDirectory() as d:
            models = FakeModels()
            pipe = Pipeline(models, CONFIG, d, lambda *a: None, threading.Event(), Empty())
            pipe.store.cues[1] = Cue(1, 0, 3, '仍保留')
            pipe.queue_review(pipe.store.cues[1], bytes(32000))
            pipe.reviews.close()
            pipe.review_worker()
            self.assertEqual(pipe.store.cues[1].original, '仍保留')
            self.assertIn('未识别', pipe.store.cues[1].review_status)
            self.assertEqual(models.calls, [])

    def test_request_limit_and_stale_revision_do_not_send(self):
        with tempfile.TemporaryDirectory() as d:
            cloud = FakeCloud()
            cfg = {**CONFIG, 'cloud': {**CONFIG['cloud'], 'max_requests_per_session': 1}}
            pipe = Pipeline(FakeModels(), cfg, d, lambda *a: None, threading.Event(), cloud)
            pipe.store.cues[1] = Cue(1, 0, 3, '已更新', revision=2)
            pipe.reviews.put({'id':1, 'revision':1, 'pcm':bytes(32000)})
            pipe.reviews.close()
            pipe.review_worker()
            self.assertEqual(cloud.calls, [])
            pipe.review_requests = 1
            pipe.queue_review(pipe.store.cues[1], bytes(32000))
            self.assertIn('请求上限', pipe.store.cues[1].review_status)


if __name__ == '__main__':
    unittest.main()
