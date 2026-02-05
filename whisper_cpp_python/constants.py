import os
from pathlib import Path

_home = Path("~").expanduser()
_xdg_data_home = os.environ.get('XDG_DATA_HOME')

xdg_data_home = (
    Path(_xdg_data_home) if _xdg_data_home else _home / ".local" / "share"
)

# MODELS URL MODELS_BASE_URL+ '/' + MODELS_PREFIX_URL+'-'+MODEL_NAME+'.bin'
# example = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin"
MODELS_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp"
MODELS_PREFIX_URL = "resolve/main/ggml"

MODELS_DIR = xdg_data_home / "whisper-cpp-python" / 'models'

AVAILABLE_MODELS = [
    "base",
    "base-q5_1",
    "base-q8_0",
    "base.en",
    "base.en-q5_1",
    "base.en-q8_0",
    "large-v1",
    "large-v2",
    "large-v2-q5_0",
    "large-v2-q8_0",
    "large-v3",
    "large-v3-q5_0",
    "large-v3-turbo",
    "large-v3-turbo-q5_0",
    "large-v3-turbo-q8_0",
    "medium",
    "medium-q5_0",
    "medium-q8_0",
    "medium.en",
    "medium.en-q5_0",
    "medium.en-q8_0",
    "small",
    "small-q5_1",
    "small-q8_0",
    "small.en",
    "small.en-q5_1",
    "small.en-q8_0",
    "tiny",
    "tiny-q5_1",
    "tiny-q8_0",
    "tiny.en",
    "tiny.en-q5_1",
    "tiny.en-q8_0",
]
