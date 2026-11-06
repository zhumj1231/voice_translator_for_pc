import io
import json
from pathlib import Path
import ssl
import tempfile
import threading
import unittest
from unittest import mock

from backends import CloudReviewer
from engine import Cue, Pipeline
from test_pipeline import CONFIG, FakeModels
import xai_stt as xai

RESULT = {'text':'Bonjour.', 'language':'fr', 'duration':3,
          'words':[{'text':'Bonjour.', 'start':.4, 'end':2.8, 'speaker':0}]}


class XaiTests(unittest.TestCase):
    def test_multipart_file_last_and_auto_does_not_request_invalid_formatting(self):
        parts, headers = xai.multipart(b'RIFF-test', xai.options({'keyterms':['赵云','Grok']}))
        body = b''.join(parts)
        self.assertEqual(int(headers['Content-Length']),len(body))
        self.assertNotIn(b'name="language"',body)
        self.assertIn(b'name="format"\r\n\r\nfalse',body)
        self.assertEqual(body.count(b'name="keyterm"'),2)
        self.assertLess(body.rindex(b'name="keyterm"'),body.index(b'name="file"'))
        forced = dict(xai.options({'force_lang':'zh','filler_words':True}))
        self.assertEqual((forced['format'],forced['language'],forced['filler_words']),('true','zh','true'))
        with self.assertRaises(ValueError): xai.options({'keyterms':['x']*101})
        with self.assertRaises(ValueError): xai.options({'keyterms':['x'*51]})

    def test_file_is_streamed_and_size_checked_before_upload(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'音频.mp3'
            path.write_bytes(b'mp3data')
            parts, headers = xai.multipart(path,xai.options({}))
            self.assertEqual(len(b''.join(parts)),int(headers['Content-Length']))
            with mock.patch('xai_stt.MAX_BYTES',2), self.assertRaises(ValueError):
                xai.multipart(path,[])

    def test_xai_never_reads_gemini_credential(self):
        with mock.patch.dict('os.environ',{},clear=True), mock.patch('backends.get_key') as gemini:
            reviewer = CloudReviewer({'enabled':True,'provider':'xai'})
            self.assertFalse(reviewer.available)
            with self.assertRaisesRegex(RuntimeError,'XAI_API_KEY'):
                reviewer.review(bytes(32000),'ja')
            gemini.assert_not_called()

    @mock.patch.dict('os.environ',{'XAI_API_KEY':'test-only-xai'})
    def test_transport_auth_tls_and_response_mapping(self):
        connection = mock.MagicMock()
        response = connection.getresponse.return_value
        response.status, response.read.return_value = 200, json.dumps(RESULT).encode()
        tls = mock.Mock()
        with (mock.patch('xai_stt.http.client.HTTPConnection',return_value=connection),
              mock.patch('xai_stt.ssl.create_default_context',return_value=tls)):
            result = xai.transcribe_once(b'wav',{'proxy_url':'http://127.0.0.1:7890'},3)
            self.assertEqual(result['language'],'fr')
            self.assertEqual(result['words'][0]['speaker'],0)
            connection.set_tunnel.assert_called_once_with('api.x.ai',443)
            self.assertEqual(connection.request.call_args.kwargs['headers']['Authorization'],'Bearer test-only-xai')
            self.assertEqual(tls.wrap_socket.call_args.kwargs['server_hostname'],'api.x.ai')
        connection.close.assert_called_once()

    @mock.patch.dict('os.environ',{'XAI_API_KEY':'test-only-xai'})
    def test_failures_never_follow_redirect_or_retry_timeout(self):
        for status, expected in [(401,'密钥'),(402,'余额'),(413,'音频过大'),(429,'429'),(302,'302')]:
            with self.subTest(status=status):
                connection = mock.MagicMock()
                connection.getresponse.return_value.status = status
                connection.getresponse.return_value.getheader.return_value = '2'
                with mock.patch('xai_stt.http.client.HTTPSConnection',return_value=connection):
                    with self.assertRaisesRegex(RuntimeError,expected):
                        xai.transcribe_once(b'wav',{})
                    self.assertEqual(connection.request.call_count,1)
        connection = mock.MagicMock()
        connection.connect.side_effect = TimeoutError()
        with mock.patch('xai_stt.http.client.HTTPSConnection',return_value=connection):
            with self.assertRaisesRegex(RuntimeError,'超时'):
                xai.transcribe_once(b'wav',{})
        connection.request.assert_not_called()
        with self.assertRaises(ValueError):
            xai.transcribe_once(b'wav',{'endpoint':'https://example.com/v1'})

    @mock.patch.dict('os.environ',{'XAI_API_KEY':'test-only-xai'})
    def test_certificate_failure_never_sends_bearer(self):
        connection, tls = mock.MagicMock(), mock.Mock()
        tls.wrap_socket.side_effect = ssl.SSLError()
        with (mock.patch('xai_stt.http.client.HTTPConnection',return_value=connection),
              mock.patch('xai_stt.ssl.create_default_context',return_value=tls)):
            with self.assertRaisesRegex(RuntimeError,'TLS'):
                xai.transcribe_once(b'wav',{'proxy_url':'http://127.0.0.1:7890'})
        connection.request.assert_not_called()

    def test_bad_timestamp_or_duration_rejected(self):
        for data in ({**RESULT,'duration':float('nan')}, {**RESULT,'words':[{'text':'bad','start':0,'end':40}]}):
            with self.assertRaises(RuntimeError): xai.normalize(data)
        empty = xai.normalize({'text':'','language':'','duration':3})
        self.assertEqual(empty['words'],[])

    def test_cloud_metadata_uses_uploaded_pcm_origin_not_vad_start(self):
        class Cloud:
            available = True
            last_transcription = RESULT
            def review(self,*a): return RESULT['text']
        with tempfile.TemporaryDirectory() as folder:
            pipe = Pipeline(FakeModels(),CONFIG,folder,lambda *a:None,threading.Event(),Cloud())
            cue = pipe.store.cues[1] = Cue(1,101,103,'本地原文',detected_language='ja',detected_language_source='qwen3')
            pipe.queue_review(cue,bytes(3*32000),audio_start=100)
            pipe.reviews.close()
            pipe.review_worker()
            self.assertEqual(cue.review_original,'Bonjour.')
            self.assertEqual(cue.review_language,'fr')
            self.assertEqual(cue.review_words[0]['start'],100.4)
            self.assertEqual(cue.original,'本地原文')
            self.assertEqual(cue.detected_language,'ja')
            self.assertTrue(cue.review_translation)
            pipe.export()
            self.assertEqual(json.loads((Path(folder)/'captions.json').read_text(encoding='utf-8'))[0]['review_words'][0]['end'],102.8)

    def test_429_retries_count_against_request_and_audio_limits(self):
        class Cloud:
            available = True
            def __init__(self): self.calls=0
            def review(self,*a):
                self.calls+=1
                raise xai.RateLimited()
        for limits, attempts in [({'max_requests_per_session':2},2),({'max_audio_seconds_per_session':3},1)]:
            with self.subTest(limits=limits), tempfile.TemporaryDirectory() as folder:
                cloud, stop = Cloud(), threading.Event()
                config = {**CONFIG,'cloud':{**CONFIG['cloud'],'provider':'xai',**limits}}
                pipe = Pipeline(FakeModels(),config,folder,lambda *a:None,stop,cloud)
                cue = pipe.store.cues[1] = Cue(1,0,3,'原文')
                pipe.queue_review(cue,bytes(3*32000))
                pipe.reviews.close()
                with mock.patch.object(stop,'wait',return_value=False): pipe.review_worker()
                self.assertEqual(cloud.calls,attempts)
                self.assertEqual(pipe.review_requests,attempts)
                self.assertEqual(pipe.review_seconds,3*attempts)
                self.assertEqual(cue.original,'原文')
                self.assertIn('上限',cue.review_status)

    def test_stopping_during_backoff_prevents_next_upload(self):
        cloud = mock.Mock()
        cloud.available = True
        cloud.review.side_effect = xai.RateLimited()
        with tempfile.TemporaryDirectory() as folder:
            stop = threading.Event()
            config = {**CONFIG,'cloud':{**CONFIG['cloud'],'provider':'xai'}}
            pipe = Pipeline(FakeModels(),config,folder,lambda *a:None,stop,cloud)
            pipe.store.cues[1] = Cue(1,0,3,'原文')
            pipe.queue_review(pipe.store.cues[1],bytes(32000))
            pipe.reviews.close()
            with mock.patch.object(stop,'wait',return_value=True): pipe.review_worker()
            cloud.review.assert_called_once()
            self.assertEqual(pipe.review_failures,0)


if __name__ == '__main__': unittest.main()
