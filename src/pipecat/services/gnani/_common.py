#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared constants and helpers for Gnani Vachana Pipecat services.

Re-exports from the ``pipecat-gnani`` package. All constants, language maps,
and helper functions originate there to keep a single source of truth.

API docs: https://docs.gnani.ai/api/introduction/introduction
"""

from pipecat_gnani._common import (
    GNANI_STT_REST_URL,
    GNANI_STT_WS_URL,
    GNANI_TTS_REST_URL,
    GNANI_TTS_SSE_URL,
    GNANI_TTS_WS_URL,
    STT_FORMAT_TRANSCRIBE,
    STT_FORMAT_VERBATIM,
    STT_LANGUAGE_MAP,
    STT_SUPPORTED_FORMATS,
    STT_SUPPORTED_SAMPLE_RATES,
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
    "STT_FORMAT_TRANSCRIBE",
    "STT_FORMAT_VERBATIM",
    "STT_LANGUAGE_MAP",
    "STT_SUPPORTED_FORMATS",
    "STT_SUPPORTED_SAMPLE_RATES",
    "SUPPORTED_VOICES",
    "TTS_LANGUAGE_MAP",
    "TTS_SUPPORTED_SAMPLE_RATES",
    "get_language_string",
    "stt_language_to_gnani",
    "tts_language_to_gnani",
]
