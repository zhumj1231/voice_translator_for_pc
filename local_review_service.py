"""CPU-only secondary ASR in an isolated process. No upload or reference text."""
import base64
import json
import sys
import time
import numpy as np
from paths import SENSEVOICE
import sherpa_onnx


def send(value):
    print(json.dumps(value,ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:
        path=SENSEVOICE
        model=sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(path/'model.int8.onnx'),tokens=str(path/'tokens.txt'),
            num_threads=2,provider='cpu',language='auto',use_itn=True)
        send({'ready':True})
        for line in sys.stdin:
            data=json.loads(line)
            pcm=base64.b64decode(data['pcm'])
            samples=np.frombuffer(pcm,'<i2').astype(np.float32)/32768
            t=time.perf_counter()
            stream=model.create_stream()
            stream.accept_waveform(16000,samples)
            model.decode_stream(stream)
            result=json.loads(str(stream.result))
            send({'text':result['text'],'language':result.get('lang',''),
                  'audio_event':result.get('event',''),'seconds':round(time.perf_counter()-t,3)})
    except Exception as error:
        send({'error':str(error)})
        raise
