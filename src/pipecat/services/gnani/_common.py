#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared constants and helpers re-exported from ``pipecat-gnani``.

All constants, language maps, and helper functions originate in
``pipecat_gnani._common``; this module keeps them accessible under
``pipecat.services.gnani._common`` for backward compatibility.
"""

from pipecat_gnani._common import (  # noqa: F401
    GNANI_STT_REST_URL,
    GNANI_STT_WS_URL,
    GNANI_TTS_REST_URL,
    GNANI_TTS_SSE_URL,
    GNANI_TTS_WS_URL,
    STREAM_CHUNK_BYTES,
    STREAM_SUPPORTED_LANGUAGES,
    STT_FORMAT_TRANSCRIBE,
    STT_FORMAT_VERBATIM,
    STT_LANGUAGE_MAP,
    STT_SUPPORTED_FORMATS,
    STT_SUPPORTED_SAMPLE_RATES,
    STT_SUPPORTED_LANGUAGES,
    SUPPORTED_BITRATES,
    SUPPORTED_CONTAINERS,
    SUPPORTED_ENCODINGS,
    SUPPORTED_MODELS,
    SUPPORTED_VOICES,
    TTS_LANGUAGE_MAP,
    TTS_SUPPORTED_SAMPLE_RATES,
    get_language_string,
    stt_language_to_gnani,
    tts_language_to_gnani,
)

__all__ = [
    "GNANI_STT_REST_URL",
    "GNANI_STT_WS_URL",
    "GNANI_TTS_REST_URL",
    "GNANI_TTS_SSE_URL",
    "GNANI_TTS_WS_URL",
    "STREAM_CHUNK_BYTES",
    "STREAM_SUPPORTED_LANGUAGES",
    "STT_FORMAT_TRANSCRIBE",
    "STT_FORMAT_VERBATIM",
    "STT_LANGUAGE_MAP",
    "STT_SUPPORTED_FORMATS",
    "STT_SUPPORTED_SAMPLE_RATES",
    "STT_SUPPORTED_LANGUAGES",
    "SUPPORTED_BITRATES",
    "SUPPORTED_CONTAINERS",
    "SUPPORTED_ENCODINGS",
    "SUPPORTED_MODELS",
    "SUPPORTED_VOICES",
    "TTS_LANGUAGE_MAP",
    "TTS_SUPPORTED_SAMPLE_RATES",
    "get_language_string",
    "stt_language_to_gnani",
    "tts_language_to_gnani",
]
