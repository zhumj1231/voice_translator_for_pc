import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from app import detected_language_label, load_config
from backends import LocalModels
from engine import Cue, Pipeline
from test_pipeline import CONFIG, FakeModels
from test_quality import Reviewer, LOUD
from languages import native_language


class LanguageTests(unittest.TestCase):
    def test_legacy_language_cannot_force_transcription(self):
        with tempfile.TemporaryDirectory() as folder:
            models = LocalModels(Path(folder), lambda _: None, threading.Event())
            models.asr_url = 'http://127.0.0.1:1234'
            for old in ('ja', 'en', 'auto'):
                with self.subTest(old=old), mock.patch('backends.urllib.request.urlopen',
                        return_value=io.BytesIO(b'{"text":"I met Lenin in 1920."}')) as post:
                    result = models.transcribe(LOUD, old, vad=False)
                    body = post.call_args.args[0].data
                    self.assertIn(b'name="language"\r\n\r\nauto\r\n', body)
                    self.assertNotIn(b'name="language"\r\n\r\nja\r\n', body)
                    self.assertEqual(result['text'], 'I met Lenin in 1920.')

    def test_language_is_native_metadata_not_setting_or_text_guess(self):
        for code, label in [('fr', '法语'), ('en', '英语'), ('ja', '日语'), ('ko', '韩语'), ('auto', '待识别')]:
            self.assertEqual(detected_language_label({'detected_language':code,
                                                      'detected_language_source':'qwen3'}), label)
        self.assertEqual(detected_language_label({'original': 'Bonjour', 'detected_language': 'ko'}), '待识别')
        self.assertEqual(detected_language_label({'original': 'Hello', 'detected_language': 'auto'}), '待识别')

    def test_per_request_metadata_never_reuses_previous_language(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder)/'asr.log'
            line = "crispasr[qwen3]: detected '{}' (p=1.000) via model output\n"
            log.write_text(line.format('ko'), encoding='utf-8')
            models = LocalModels(Path(folder), lambda _:None, threading.Event())
            models.asr_url = 'http://127.0.0.1:1234'
            for new, expected in [(line.format('fr'), 'fr'), ('', ''), (line.format('en'), 'en')]:
                def respond(*args, **kwargs):
                    with log.open('a', encoding='utf-8') as out:
                        out.write(new)
                    return io.BytesIO(b'{"text":"primary transcript", "language":"auto"}')
                with mock.patch('backends.urllib.request.urlopen', side_effect=respond):
                    self.assertEqual(models.transcribe(LOUD,'auto',vad=False)['detected_language'], expected)
            self.assertEqual(native_language(line.format('fr') + line.format('ko')), '')
            self.assertEqual(native_language(line.format('French')), 'fr')

    def test_french_never_sent_to_five_language_reviewer(self):
        with tempfile.TemporaryDirectory() as folder:
            reviewer = mock.Mock()
            pipe = Pipeline(FakeModels(), CONFIG, folder, lambda *a:None, threading.Event(), local_reviewer=reviewer)
            cue = pipe.store.cues[1] = Cue(1,0,3,'Je crois que vous êtes amoureux.',
                                          detected_language='fr', detected_language_source='qwen3')
            pipe.queue_local_review(cue,LOUD*3)
            pipe.local_reviews.close()
            pipe.local_review_worker()
            reviewer.analyze.assert_not_called()
            self.assertEqual(detected_language_label(pipe.store.snapshot()[0]), '法语')
            self.assertEqual(cue.local_review_original, '')

    def test_secondary_language_conflict_cannot_relabel_or_translate_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            models = FakeModels('Hello there.')
            pipe = Pipeline(models, CONFIG, folder, lambda *a:None, threading.Event(),
                            local_reviewer=Reviewer('안녕하세요', '<|Speech|>', '<|ko|>'))
            cue = pipe.store.cues[1] = Cue(1,0,3,'Hello there.', detected_language='en', detected_language_source='qwen3')
            pipe.queue_local_review(cue,LOUD*3)
            pipe.local_reviews.close()
            pipe.local_review_worker()
            self.assertEqual(cue.detected_language, 'en')
            self.assertEqual(cue.local_review_original, '')
            self.assertEqual(models.calls, [])

    def test_pipeline_exports_primary_french_and_skips_unsupported_review(self):
        with tempfile.TemporaryDirectory() as folder:
            class FrenchModel(FakeModels):
                def transcribe(self, *args, **kwargs):
                    result = super().transcribe(*args, **kwargs)
                    result.update(detected_language='fr', detected_language_source='qwen3')
                    return result
            reviewer = mock.Mock()
            pipe = Pipeline(FrenchModel('Je crois que vous êtes amoureux.'), CONFIG, folder,
                            lambda *a:None, threading.Event(), local_reviewer=reviewer)
            pipe.audio.append(LOUD*3)
            pipe.audio.finish()
            pipe.run()
            saved = json.loads((Path(folder)/'captions.json').read_text(encoding='utf-8'))[0]
            self.assertEqual(detected_language_label(saved), '法语')
            self.assertEqual(saved['original'], 'Je crois que vous êtes amoureux.')
            self.assertTrue(saved['translation'])
            reviewer.analyze.assert_not_called()

    def test_old_config_migrates_without_influencing_output(self):
        config = load_config()
        config['language'] = 'ja'
        with mock.patch('pathlib.Path.read_text', return_value=json.dumps(config)):
            self.assertEqual(load_config()['language'], 'auto')

    def test_translation_always_requests_chinese_for_both_languages(self):
        with tempfile.TemporaryDirectory() as folder:
            models = LocalModels(Path(folder), lambda _: None, threading.Event())
            models.translation_url = 'http://127.0.0.1:1234'
            for source in ('I met Lenin in 1920.', 'こんにちは。'):
                with mock.patch('backends.post_json', return_value={
                        'choices': [{'message': {'content': '中文译文'}}]}) as post:
                    self.assertEqual(models.translate(source, []), '中文译文')
                    prompt = post.call_args.args[1]['messages'][0]['content']
                    self.assertIn('译文都用简体中文', prompt)
                    self.assertIn(source, prompt)


if __name__ == '__main__':
    unittest.main()
