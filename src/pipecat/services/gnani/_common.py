#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared constants and helpers for Gnani Vachana Pipecat services."""

from pipecat.transcriptions.language import Language, resolve_language

GNANI_STT_REST_URL = "https://api.vachana.ai/stt/v3"
GNANI_STT_WS_URL = "wss://api.vachana.ai/stt/v3/stream"

GNANI_TTS_REST_URL = "https://api.vachana.ai/api/v1/tts/inference"
GNANI_TTS_SSE_URL = "https://api.vachana.ai/api/v1/tts/sse"
GNANI_TTS_WS_URL = "wss://api.vachana.ai/api/v1/tts"

STT_SUPPORTED_SAMPLE_RATES = (8000, 16000)
TTS_SUPPORTED_SAMPLE_RATES = (8000, 16000, 22050, 44100)

STREAM_CHUNK_BYTES = 1024

SUPPORTED_VOICES = frozenset({
    "Karan", "Simran", "Nara", "Riya", "Viraj", "Raju",
})

SUPPORTED_LANGUAGES = frozenset({
    "as-IN", "bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN",
    "ml-IN", "mr-IN", "or-IN", "pa-IN", "ta-IN", "te-IN",
    "en-IN,hi-IN",
})

STREAM_SUPPORTED_LANGUAGES = frozenset({
    "bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN",
    "ml-IN", "mr-IN", "pa-IN", "ta-IN", "te-IN",
    "en-hi-IN-latn", "en-hi-in-cm",
})

STT_LANGUAGE_MAP = {
    Language.AS_IN: "as-IN",
    Language.BN: "bn-IN",
    Language.BN_IN: "bn-IN",
    Language.EN: "en-IN",
    Language.EN_IN: "en-IN",
    Language.GU: "gu-IN",
    Language.GU_IN: "gu-IN",
    Language.HI: "hi-IN",
    Language.HI_IN: "hi-IN",
    Language.KN: "kn-IN",
    Language.KN_IN: "kn-IN",
    Language.ML: "ml-IN",
    Language.ML_IN: "ml-IN",
    Language.MR: "mr-IN",
    Language.MR_IN: "mr-IN",
    Language.OR: "or-IN",
    Language.OR_IN: "or-IN",
    Language.PA: "pa-IN",
    Language.PA_IN: "pa-IN",
    Language.TA: "ta-IN",
    Language.TA_IN: "ta-IN",
    Language.TE: "te-IN",
    Language.TE_IN: "te-IN",
}

TTS_LANGUAGE_MAP = {
    Language.AS: "as-IN",
    Language.AS_IN: "as-IN",
    Language.BN: "bn-IN",
    Language.BN_IN: "bn-IN",
    Language.EN: "en-IN",
    Language.EN_IN: "en-IN",
    Language.GU: "gu-IN",
    Language.GU_IN: "gu-IN",
    Language.HI: "hi-IN",
    Language.HI_IN: "hi-IN",
    Language.KN: "kn-IN",
    Language.KN_IN: "kn-IN",
    Language.ML: "ml-IN",
    Language.ML_IN: "ml-IN",
    Language.MR: "mr-IN",
    Language.MR_IN: "mr-IN",
    Language.OR: "or-IN",
    Language.OR_IN: "or-IN",
    Language.PA: "pa-IN",
    Language.PA_IN: "pa-IN",
    Language.TA: "ta-IN",
    Language.TA_IN: "ta-IN",
    Language.TE: "te-IN",
    Language.TE_IN: "te-IN",
}


def stt_language_to_gnani(language: Language) -> str:
    """Convert a Language enum to Gnani STT language code."""
    return resolve_language(language, STT_LANGUAGE_MAP, use_base_code=False)


def tts_language_to_gnani(language: Language) -> str | None:
    """Convert a Language enum to Gnani TTS language code."""
    return resolve_language(language, TTS_LANGUAGE_MAP, use_base_code=False)


def get_language_string(settings, converter) -> str | None:
    """Resolve the language setting to a string code."""
    if settings.language:
        if isinstance(settings.language, Language):
            return converter(settings.language)
        return str(settings.language)
    return "en-IN"
