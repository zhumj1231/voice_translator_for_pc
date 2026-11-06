"""Read-only readiness check. Never starts models, captures audio, or downloads."""
import importlib.util
import json
import os
import sys
from paths import (config_path, CRISPASR, LLAMA_SERVER, ASR_MODEL, TRANSLATION_MODEL,
                   VAD_MODEL, SENSEVOICE, SPEAKER_MODELS)


def check():
    config = json.loads(config_path().read_text(encoding='utf-8-sig'))
    failures = []
    if os.name != 'nt' or sys.version_info[:2] != (3, 12) or sys.maxsize <= 2**32:
        failures.append('Windows x64 / Python 3.12 required')
    modules = ['tkinter', 'numpy', 'pyaudiowpatch']
    if config.get('speakers', {}).get('enabled') or config.get('local_review', {}).get('enabled'):
        modules.append('sherpa_onnx')
    for name in modules:
        if importlib.util.find_spec(name) is None:
            failures.append('Missing Python package: ' + name)
    required = [CRISPASR, LLAMA_SERVER, ASR_MODEL, TRANSLATION_MODEL]
    if config.get('vad'):
        required.append(VAD_MODEL)
    if config.get('local_review', {}).get('enabled'):
        required.extend([SENSEVOICE/'model.int8.onnx', SENSEVOICE/'tokens.txt'])
    if config.get('speakers', {}).get('enabled'):
        required.extend([SPEAKER_MODELS/'sherpa-onnx-pyannote-segmentation-3-0/model.onnx',
                         SPEAKER_MODELS/'3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx'])
    for path in required:
        if not path.is_file():
            failures.append('Missing resource: ' + str(path))
    for problem in failures:
        print('[MISSING] ' + problem)
    print('Cloud provider: ' + config['cloud']['provider'] + '; enabled: ' + str(config['cloud']['enabled']))
    if config['cloud']['enabled']:
        from backends import CloudReviewer
        print('Cloud key configured: ' + str(CloudReviewer(config['cloud']).available))
    print('Check passed (existence only; GPU/model inference not tested).' if not failures else
          'Setup incomplete. Follow docs/DEPLOYMENT.md; no downloads were started.')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(check())
