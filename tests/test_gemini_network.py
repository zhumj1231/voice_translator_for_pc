import json
import ssl
import unittest
from unittest import mock
from gemini_network import HOST, verified_post


class ProxyTransportTests(unittest.TestCase):
    def test_controller_and_mode_not_required_for_success(self):
        connection = mock.MagicMock()
        response = connection.getresponse.return_value
        response.status = 200
        response.read.return_value = b'{"status":"completed"}'
        tls = mock.Mock()
        with (mock.patch('pathlib.Path.read_text', side_effect=AssertionError('must not read proxy config')),
              mock.patch('gemini_network.http.client.HTTPConnection', return_value=connection),
              mock.patch('gemini_network.ssl.create_default_context', return_value=tls)):
            result = verified_post('/v1beta/interactions', {'input':'test-audio'}, 'test-api-key', 'http://127.0.0.1:7890')
        self.assertEqual(result['status'], 'completed')
        connection.set_tunnel.assert_called_once_with(HOST, 443)
        self.assertEqual(tls.wrap_socket.call_args.kwargs['server_hostname'], HOST)
        connection.request.assert_called_once()
        connection.close.assert_called_once()

    def test_tls_failure_never_sends_key_or_audio(self):
        connection = mock.MagicMock()
        tls = mock.Mock()
        tls.wrap_socket.side_effect = ssl.SSLError('test-certificate-error')
        with (mock.patch('gemini_network.http.client.HTTPConnection', return_value=connection),
              mock.patch('gemini_network.ssl.create_default_context', return_value=tls)):
            with self.assertRaisesRegex(RuntimeError, 'TLS'):
                verified_post('/v1beta/interactions', {'input':'test-audio'}, 'test-api-key', 'http://127.0.0.1:7890')
        connection.request.assert_not_called()
        connection.close.assert_called_once()

    def test_timeout_explains_stage_without_automatic_retry(self):
        connection = mock.MagicMock()
        connection.connect.side_effect = TimeoutError('test')
        with mock.patch('gemini_network.http.client.HTTPConnection', return_value=connection):
            with self.assertRaisesRegex(RuntimeError, '连接本地代理超时'):
                verified_post('/v1beta/interactions', {}, 'test-api-key', 'http://127.0.0.1:7890')
        connection.connect.assert_called_once()
        connection.request.assert_not_called()

    def test_redirect_not_followed(self):
        connection = mock.MagicMock()
        connection.getresponse.return_value.status = 302
        connection.getresponse.return_value.read.return_value = b''
        with (mock.patch('gemini_network.http.client.HTTPConnection', return_value=connection),
              mock.patch('gemini_network.ssl.create_default_context')):
            with self.assertRaisesRegex(RuntimeError, 'HTTP 302'):
                verified_post('/v1beta/interactions', {}, 'test-api-key', 'http://127.0.0.1:7890')
        connection.request.assert_called_once()

    def test_api_error_detail_is_visible_and_key_is_redacted(self):
        connection = mock.MagicMock()
        connection.getresponse.return_value.status = 400
        connection.getresponse.return_value.read.return_value = json.dumps({
            'error':{'message':'invalid test-api-key'}}).encode()
        with (mock.patch('gemini_network.http.client.HTTPConnection', return_value=connection),
              mock.patch('gemini_network.ssl.create_default_context')):
            with self.assertRaises(RuntimeError) as caught:
                verified_post('/v1beta/interactions', {}, 'test-api-key', 'http://127.0.0.1:7890')
        self.assertIn('HTTP 400', str(caught.exception))
        self.assertIn('已隐藏密钥', str(caught.exception))
        self.assertNotIn('test-api-key', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
