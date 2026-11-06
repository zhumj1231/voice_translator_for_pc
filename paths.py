"""Portable resource locations; no downloads or machine-specific paths."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def location(variable, default):
    value = Path(os.environ.get(variable) or default).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


DATA = location('VOICE_TRANSLATOR_DATA_DIR', ROOT / 'data')
TOOLS = location('VOICE_TRANSLATOR_TOOLS_DIR', ROOT / 'assets' / 'tools')
SPEAKER_MODELS = location('VOICE_TRANSLATOR_SPEAKER_MODELS_DIR', ROOT / 'assets' / 'speaker-models')
REVIEW_MODELS = location('VOICE_TRANSLATOR_REVIEW_MODELS_DIR', ROOT / 'assets' / 'review-models')
CRISPASR = TOOLS / 'crispasr' / 'crispasr-windows-x86_64-vulkan' / 'crispasr.exe'
LLAMA_SERVER = TOOLS / 'llama' / 'llama-server.exe'
ASR_MODEL = TOOLS / 'models' / 'qwen3-asr-1.7b-q8_0.gguf'
TRANSLATION_MODEL = TOOLS / 'models' / 'HY-MT2-7B-Q8_0.gguf'
VAD_MODEL = TOOLS / 'models' / 'ggml-silero-v6.2.0.bin'
SENSEVOICE = REVIEW_MODELS / 'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17'


def config_path():
    custom = ROOT / 'config.json'
    return custom if custom.is_file() else ROOT / 'config.example.json'
