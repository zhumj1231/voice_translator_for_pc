import unittest
from unittest import mock
from backends import CloudReviewer, gemini_post, _NoCloudRedirect


class GeminiTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {'enabled': True, 'provider': 'gemini_interactions',
                    'model': 'gemini-3.5-transcribe',
                    'endpoint': 'https://generativelanguage.googleapis.com/v1beta'}

    @mock.patch('backends.get_key', return_value='test-only')
    def test_audio_only_verbatim_and_output_not_thought(self, key):
        response = {'status': 'completed', 'steps': [
            {'type': 'thought', 'content': [{'type': 'text', 'text': 'ignore'}]},
            {'type': 'model_output', 'content': [{'type': 'text', 'text': 'あの、僕は'}]}]}
        with mock.patch('backends.gemini_post', return_value=response) as post:
            self.assertEqual(CloudReviewer(self.cfg).review(bytes(32000), 'ja'), 'あの、僕は')
            payload = post.call_args.args[1]
            self.assertEqual([x['type'] for x in payload['input']], ['audio'])
            self.assertEqual(payload['generation_config']['transcription_config'],
                             {'language_codes': [], 'mode': {'type': 'verbatim'}})
            self.assertFalse(payload['store'])
            self.assertEqual(post.call_count, 1)

    @mock.patch('backends.get_key', return_value='test-only')
    def test_auto_does_not_force_english_or_japanese(self, key):
        with mock.patch('backends.gemini_post', return_value={'status':'completed','steps':[]}) as post:
            CloudReviewer(self.cfg).review(bytes(32000), 'auto')
            self.assertEqual(post.call_args.args[1]['generation_config']['transcription_config']['language_codes'], [])

    @mock.patch('backends.get_key', return_value='test-only')
    def test_incomplete_not_accepted_and_oversized_not_sent(self, key):
        with mock.patch('backends.gemini_post', return_value={'status': 'incomplete'}) as post:
            with self.assertRaises(RuntimeError):
                CloudReviewer(self.cfg).review(bytes(32000), 'ja')
            with self.assertRaises(ValueError):
                CloudReviewer(self.cfg).review(bytes(31 * 32000), 'ja')
            self.assertEqual(post.call_count, 1)

    def test_key_never_sent_to_other_endpoint(self):
        with mock.patch('urllib.request.build_opener') as opener:
            for url in ['https://example.com/interactions',
                        'http://generativelanguage.googleapis.com/interactions',
                        'https://generativelanguage.googleapis.com.evil.invalid/interactions']:
                with self.assertRaises(ValueError):
                    gemini_post(url, {}, 'test-only')
            opener.assert_not_called()
        with self.assertRaises(RuntimeError):
            _NoCloudRedirect().redirect_request(None, None, 302, '', {}, 'https://example.com')


if __name__ == '__main__':
    unittest.main()
