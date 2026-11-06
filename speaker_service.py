"""Private stdio worker. Audio stays on this computer; stdout is JSON only."""
import base64
import json
import sys
import time
from speakers import SpeakerAnalyzer


def send(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        analyzer = SpeakerAnalyzer()
        send({'ready': True})
        for line in sys.stdin:
            request = json.loads(line)
            started = time.perf_counter()
            analyzer.registry.match = float(request.get('match', .4))
            result = analyzer.analyze(base64.b64decode(request['pcm']), float(request['start']), request.get('closed',False))
            result['seconds'] = round(time.perf_counter()-started,3)
            send(result)
    except Exception as error:
        send({'error': str(error)})
        raise
