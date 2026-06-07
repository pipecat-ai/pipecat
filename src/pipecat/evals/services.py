#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Service constructors for the eval harness.

Each function builds a concrete pipecat service (TTS, STT, or judge LLM) from a
scenario's config mapping. They are the dispatch targets behind the ``service:``
name in :meth:`pipecat.evals.voice.EvalVoice.from_config`,
:meth:`pipecat.evals.transcribe.EvalTranscriber.from_config`, and
:meth:`pipecat.evals.judge.EvalJudge.from_config`. The heavy provider imports
stay lazy inside each function so importing this module stays cheap.
"""

import os
from typing import Any

from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService


def kokoro_service(voice_cfg: dict, sample_rate: int) -> TTSService:
    """Build a local Kokoro TTS service from the ``user_audio`` config.

    Kokoro runs an ONNX model locally (no API key, no per-run cost), so the eval
    suite synthesizes user audio for free. The model files are downloaded once
    on first use and cached under ``~/.cache/kokoro-onnx``.
    """
    from pipecat.services.kokoro.tts import KokoroTTSService

    return KokoroTTSService(
        settings=KokoroTTSService.Settings(voice=str(voice_cfg.get("voice", ""))),
        sample_rate=sample_rate,
    )


def cartesia_service(voice_cfg: dict, sample_rate: int) -> TTSService:
    """Build a Cartesia TTS service from the ``user_audio`` config."""
    from pipecat.services.cartesia.tts import CartesiaHttpTTSService

    # Prefer an explicit api_key in the config; fall back to the env var so
    # committed scenarios don't carry secrets.
    api_key = voice_cfg.get("api_key") or os.environ.get("CARTESIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Cartesia API key not found — set $CARTESIA_API_KEY or user_audio.api_key"
        )

    return CartesiaHttpTTSService(
        api_key=api_key,
        settings=CartesiaHttpTTSService.Settings(
            voice=str(voice_cfg.get("voice", "")),
            model=voice_cfg.get("model") or "sonic-2",
        ),
        sample_rate=sample_rate,
    )


def whisper_service(config: dict) -> STTService:
    """Build a local Whisper STT service from the ``bot_audio`` config.

    The eval transcribes audio it already knows is the bot speaking (the harness
    captures it between ``bot-started-speaking`` and ``bot-stopped-speaking``), so
    Whisper's non-speech filter is counterproductive here: the default
    ``no_speech_prob=0.4`` drops correct transcriptions of synthetic/TTS speech,
    whose ``no_speech_prob`` jitters across ~0.4-0.6 run to run (a dropped segment
    yields no ``TranscriptionFrame``, so the harness then waits out the whole
    transcription timeout). Disable the filter with a permissive threshold.
    """
    from pipecat.services.whisper.stt import WhisperSTTService

    kwargs: dict = {"no_speech_prob": 1.0}
    model = config.get("model")
    if model:
        kwargs["model"] = str(model)
    return WhisperSTTService(settings=WhisperSTTService.Settings(**kwargs))


def ollama_service(config: dict) -> LLMService[Any]:
    """Build a local Ollama LLM service from the ``judge:`` config."""
    from pipecat.services.ollama.llm import OLLamaLLMService

    base_url = config.get("endpoint") or "http://localhost:11434/v1"
    return OLLamaLLMService(
        base_url=base_url,
        settings=OLLamaLLMService.Settings(model=config.get("model", "qwen2.5:3b")),
    )


def openai_service(config: dict) -> LLMService[Any]:
    """Build an OpenAI LLM service from the ``judge:`` config."""
    from pipecat.services.openai.llm import OpenAILLMService

    return OpenAILLMService(settings=OpenAILLMService.Settings(model=config.get("model", "gpt-4o")))
