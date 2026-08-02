#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Service constructors for the eval harness.

Each function builds a concrete pipecat service (TTS, STT, or judge LLM) from a
scenario's config mapping. They are the dispatch targets behind the ``service:``
name in :meth:`pipecat.evals.speech.EvalSpeech.from_config`,
:meth:`pipecat.evals.transcribe.EvalTranscriber.from_config`, and
:meth:`pipecat.evals.judge.EvalJudge.from_config`. The heavy provider imports
stay lazy inside each function so importing this module stays cheap.
"""

import os
from typing import Any

from pipecat.services.llm_service import LLMService
from pipecat.services.settings import NOT_GIVEN
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language


def _cfg_language(cfg: dict) -> Language | None:
    """Coerce a config's optional ``language`` value to a :class:`Language`.

    ``Language`` is a ``StrEnum``, so both a code string (e.g. ``"zh"``) and a
    ``Language`` are accepted. Each concrete service maps the ``Language`` to its
    own provider code internally (via ``resolve_language``), so the eval layer
    only needs to hand off a ``Language``.

    Args:
        cfg: A ``user.speech`` or ``judge.transcription`` config mapping.

    Returns:
        The resolved ``Language``, or ``None`` when ``language`` is absent (so
        callers leave the service's own default untouched).

    Raises:
        ValueError: If ``language`` is set to a value that is not a recognized
            language code.
    """
    value = cfg.get("language")
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        return Language(value)
    except ValueError as e:
        raise ValueError(
            f"Unknown language {value!r} in eval speech/transcription config; "
            "expected a language code like 'zh' or a Language value."
        ) from e


def kokoro_service(voice_cfg: dict, sample_rate: int) -> TTSService:
    """Build a local Kokoro TTS service from the ``user_audio`` config.

    Kokoro runs an ONNX model locally (no API key, no per-run cost), so the eval
    suite synthesizes user audio for free. The model files are downloaded once
    on first use and cached under ``~/.cache/kokoro-onnx``.

    Config keys:
        voice: Kokoro voice id (e.g. ``af_heart``).
        language: Optional language code (e.g. ``zh``) or ``Language`` value.
            When omitted, Kokoro keeps its own default.
    """
    from pipecat.services.kokoro.tts import KokoroTTSService

    settings: dict[str, Any] = {"voice": str(voice_cfg.get("voice", ""))}
    language = _cfg_language(voice_cfg)
    if language is not None:
        settings["language"] = language

    return KokoroTTSService(
        settings=KokoroTTSService.Settings(**settings),
        sample_rate=sample_rate,
    )


def cartesia_service(voice_cfg: dict, sample_rate: int) -> TTSService:
    """Build a Cartesia TTS service from the ``user_audio`` config.

    Config keys:
        voice: Cartesia voice id.
        model: Optional model (defaults to ``sonic-2``).
        api_key: Optional key (falls back to ``$CARTESIA_API_KEY``).
        language: Optional language code (e.g. ``zh``) or ``Language`` value.
            When omitted, Cartesia keeps its own default.
    """
    from pipecat.services.cartesia.tts import CartesiaHttpTTSService

    # Prefer an explicit api_key in the config; fall back to the env var so
    # committed scenarios don't carry secrets.
    api_key = voice_cfg.get("api_key") or os.environ.get("CARTESIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Cartesia API key not found — set $CARTESIA_API_KEY or user_audio.api_key"
        )

    settings: dict[str, Any] = {
        "voice": str(voice_cfg.get("voice", "")),
        "model": voice_cfg.get("model") or "sonic-2",
    }
    language = _cfg_language(voice_cfg)
    if language is not None:
        settings["language"] = language

    return CartesiaHttpTTSService(
        api_key=api_key,
        settings=CartesiaHttpTTSService.Settings(**settings),
        sample_rate=sample_rate,
    )


def whisper_service(config: dict) -> STTService:
    """Build a local Whisper STT service from the ``bot_audio`` config.

    Runs on the **CPU** by default (``device: cpu``): the GPU is reserved for the
    judge LLM and the per-run audio models, and bot-speech transcription happens
    once per turn off the hot path, so the extra latency is fine. This frees enough
    GPU memory to run a larger, more accurate model (e.g. ``distil-medium`` or
    ``large-v3-turbo``) at higher concurrency. Override with ``device: cuda`` (and
    ``compute_type``) in the ``transcription`` config if you have GPU headroom.

    The eval transcribes audio it already knows is the bot speaking (the harness
    captures it between ``bot-started-speaking`` and ``bot-stopped-speaking``), so
    Whisper's non-speech filter is counterproductive here: the default
    ``no_speech_prob=0.4`` drops correct transcriptions of synthetic/TTS speech,
    whose ``no_speech_prob`` jitters across ~0.4-0.6 run to run (a dropped segment
    yields no ``TranscriptionFrame``, so the harness then waits out the whole
    transcription timeout). Disable the filter with a permissive threshold.

    Config keys:
        device: ``cpu`` (default) or ``cuda``.
        compute_type: Whisper compute type (defaults to ``int8`` on CPU).
        model: Optional Whisper model (left unset to use Whisper's default).
        language: Optional language code (e.g. ``zh``) or ``Language`` value.
            When omitted, Whisper keeps its own default (English) — it does not
            auto-detect, so a non-English bot needs this set.
    """
    from pipecat.services.whisper.stt import WhisperSTTService

    device = config.get("device", "cpu")
    # int8 keeps CPU transcription reasonably fast with negligible accuracy loss;
    # the default ("default") would pick float32 on CPU, which is much slower.
    compute_type = config.get("compute_type", "int8" if device == "cpu" else "default")
    # NOT_GIVEN (not None) leaves the model unset so Whisper uses its own default.
    settings: dict[str, Any] = {
        "no_speech_prob": 1.0,
        "model": config.get("model", NOT_GIVEN),
    }
    language = _cfg_language(config)
    if language is not None:
        settings["language"] = language
    return WhisperSTTService(
        device=device,
        compute_type=compute_type,
        settings=WhisperSTTService.Settings(**settings),
    )


def moonshine_service(config: dict) -> STTService:
    """Build a local Moonshine STT service from the ``bot_audio`` config.

    Moonshine runs on the CPU via ONNX Runtime (no GPU, no API key) and is small
    and fast. On the short, isolated bot-answer segments the harness transcribes,
    it tends to keep the answer where Whisper sometimes drops it. ``model`` selects
    the architecture (a :class:`~pipecat.services.moonshine.stt.Model` value or
    string; default ``Model.SMALL_STREAMING``).

    Config keys:
        model: Optional architecture (a ``Model`` value or string; default
            ``Model.SMALL_STREAMING``).
        language: Optional language code (e.g. ``zh``) or ``Language`` value.
            When omitted, Moonshine keeps its own default (English). Moonshine
            supports only a handful of languages.
    """
    from pipecat.services.moonshine.stt import Model, MoonshineSTTService

    settings: dict[str, Any] = {"model": config.get("model") or Model.SMALL_STREAMING}
    language = _cfg_language(config)
    if language is not None:
        settings["language"] = language
    return MoonshineSTTService(
        settings=MoonshineSTTService.Settings(**settings),
    )


def ollama_service(config: dict) -> LLMService[Any]:
    """Build a local Ollama LLM service from the ``judge:`` config."""
    from pipecat.services.ollama.llm import OLLamaLLMService

    base_url = config.get("endpoint") or "http://localhost:11434/v1"
    return OLLamaLLMService(
        base_url=base_url,
        settings=OLLamaLLMService.Settings(model=config.get("model", "gemma2:9b")),
    )


def openai_service(config: dict) -> LLMService[Any]:
    """Build an OpenAI LLM service from the ``judge:`` config."""
    from pipecat.services.openai.llm import OpenAILLMService

    return OpenAILLMService(settings=OpenAILLMService.Settings(model=config.get("model", "gpt-4o")))
