"""Google HTTPS through a local proxy; no access to proxy controller settings."""
import http.client
import json
import ssl

HOST = 'generativelanguage.googleapis.com'


def verified_post(path, payload, key, proxy_url):
    return _verified_request('POST', path, payload, key, proxy_url)


def verified_get(path, key, proxy_url):
    """Read-only service diagnostics with the same destination and TLS checks."""
    return _verified_request('GET', path, None, key, proxy_url)


def _verified_request(method, path, payload, key, proxy_url):
    if proxy_url != 'http://127.0.0.1:7890' or not path.startswith('/v1beta/') or '?' in path:
        raise ValueError('Gemini 网络配置不受支持')
    # User policy: use the connection Clash supplies in any mode. Controller
    # availability and routing-chain labels are not prerequisites for an API call.
    # The destination and TLS certificate are still verified before sending data.
    conn = http.client.HTTPConnection('127.0.0.1', 7890, timeout=45)
    conn.set_tunnel(HOST, 443)
    phase = '连接本地代理'
    try:
        conn.connect()
        phase = '建立 Google TLS 连接'
        conn.sock = ssl.create_default_context().wrap_socket(conn.sock, server_hostname=HOST)
        phase = '发送请求或等待 Google 响应'
        conn.request(method, path, body=json.dumps(payload).encode('utf-8') if payload is not None else None,
                     headers={'Host': HOST, 'Content-Type': 'application/json', 'x-goog-api-key': key})
        response = conn.getresponse()
        if response.status != 200:
            detail = ''
            try:
                error = json.loads(response.read(16384)).get('error', {})
                message = error.get('message', '')
                if isinstance(message, str):
                    detail = ' '.join(message.replace(key, '[已隐藏密钥]').split())[:400] if key else ''
                    if 'User location is not supported for the API use' in detail:
                        detail = '当前网络出口地区不支持 Gemini API，请换用可用出口节点'
            except (ValueError, AttributeError):
                pass
            raise RuntimeError(f'Gemini HTTP {response.status}' + ('：' + detail if detail else '') + '；未重试')
        body = response.read(4*1024*1024+1)
        if len(body) > 4*1024*1024:
            raise RuntimeError('Gemini 响应过大；未自动重试')
        result = json.loads(body)
        return result
    except ssl.SSLError:
        raise RuntimeError('Gemini TLS 连接或证书验证失败；未自动重试') from None
    except TimeoutError:
        raise RuntimeError('Gemini ' + phase + '超时；未自动重试') from None
    except (OSError, http.client.HTTPException) as error:
        raise RuntimeError('Gemini ' + phase + '失败（' + type(error).__name__ + '）；未自动重试') from None
    except (ValueError, UnicodeError):
        raise RuntimeError('Gemini 响应格式异常；未自动重试') from None
    finally:
        conn.close()
