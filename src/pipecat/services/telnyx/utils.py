import os
import struct
from typing import Optional

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
TTS_WS_URL = "wss://api.telnyx.com/v2/text-to-speech/speech"
STT_WS_URL = "wss://api.telnyx.com/v2/speech-to-text/transcription"


def get_api_key(api_key: Optional[str] = None) -> str:
    key = api_key or os.environ.get("TELNYX_API_KEY")
    if not key:
        raise ValueError("TELNYX_API_KEY not provided and not found in environment")
    return key


def create_wav_header(
    sample_rate: int = DEFAULT_SAMPLE_RATE, channels: int = DEFAULT_CHANNELS
) -> bytes:
    """Create a WAV header for streaming (infinite length) at 16-bit PCM.

    This mimics the structure used in the working LiveKit implementation.
    """
    bytes_per_sample = 2
    byte_rate = sample_rate * channels * bytes_per_sample
    block_align = channels * bytes_per_sample
    data_size = 0x7FFFFFFF
    file_size = 36 + data_size

    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        file_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        16,
        b"data",
        data_size,
    )
