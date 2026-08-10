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
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.types import NOT_GIVEN, NotGiven


def _cfg_language(cfg: dict) -> Language | NotGiven:
    """Coerce a config's optional ``language`` value to a :class:`Language`.

    ``Language`` is a ``StrEnum``, so both a code string (e.g. ``"zh"``) and a
    ``Language`` are accepted. Each concrete service maps the ``Language`` to its
    own provider code internally (via ``resolve_language``), so the eval layer
    only needs to hand off a ``Language``.

    Args:
        cfg: A ``user.speech`` or ``judge.transcription`` config mapping.

    Returns:
        The resolved ``Language``, or ``NOT_GIVEN`` when ``language`` is absent,
        which leaves the service's own default in place.

    Raises:
        ValueError: If ``language`` is set to a value that is not a recognized
            language code.
    """
    value = cfg.get("language")
    if value is None:
        return NOT_GIVEN
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return NOT_GIVEN
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

    Args:
        voice_cfg: The ``user.speech`` config mapping:

            - ``voice``: Kokoro voice id (e.g. ``af_heart``).
            - ``language``: Optional language code (e.g. ``zh``) or ``Language``.
              When omitted, Kokoro keeps its own default (English). Voices are
              language-specific, so a non-English language needs a matching voice
              — ``af_heart`` speaks US English whatever the language is set to.

        sample_rate: Sample rate for the synthesized audio.
    """
    from pipecat.services.kokoro.tts import KokoroTTSService

    return KokoroTTSService(
        settings=KokoroTTSService.Settings(
            voice=str(voice_cfg.get("voice", "")),
            language=_cfg_language(voice_cfg),
        ),
        sample_rate=sample_rate,
    )


def cartesia_service(voice_cfg: dict, sample_rate: int) -> TTSService:
    """Build a Cartesia TTS service from the ``user_audio`` config.

    Args:
        voice_cfg: The ``user.speech`` config mapping:

            - ``voice``: Cartesia voice id.
            - ``model``: Optional model (defaults to ``sonic-2``).
            - ``api_key``: Optional key (falls back to ``$CARTESIA_API_KEY``).
            - ``language``: Optional language code (e.g. ``zh``) or ``Language``.
              When omitted, Cartesia keeps its own default (English).

        sample_rate: Sample rate for the synthesized audio.

    Raises:
        RuntimeError: If no API key is given in the config or the environment.
    """
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
            language=_cfg_language(voice_cfg),
        ),
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

    Args:
        config: The ``judge.transcription`` config mapping:

            - ``device``: ``cpu`` (default) or ``cuda``.
            - ``compute_type``: Whisper compute type (``int8`` on CPU).
            - ``model``: Optional Whisper model (left unset to use Whisper's own).
              Whisper's default is English-only, as is every ``.en`` model, so a
              non-English ``language`` needs a multilingual model here (e.g.
              ``large-v3-turbo``).
            - ``language``: Optional language code (e.g. ``es``) or ``Language``.
              When omitted, Whisper keeps its own default (English) — it does not
              auto-detect, so a non-English bot needs this set.

    Raises:
        ValueError: If ``language`` names a language the chosen model can't
            transcribe (raised by :class:`~pipecat.services.whisper.stt.WhisperSTTService`).
    """
    from pipecat.services.whisper.stt import WhisperSTTService

    device = config.get("device", "cpu")
    # int8 keeps CPU transcription reasonably fast with negligible accuracy loss;
    # the default ("default") would pick float32 on CPU, which is much slower.
    compute_type = config.get("compute_type", "int8" if device == "cpu" else "default")
    # NOT_GIVEN (not None) leaves the model unset so Whisper uses its own default.
    return WhisperSTTService(
        device=device,
        compute_type=compute_type,
        settings=WhisperSTTService.Settings(
            no_speech_prob=1.0,
            model=config.get("model", NOT_GIVEN),
            language=_cfg_language(config),
        ),
    )


def moonshine_service(config: dict) -> STTService:
    """Build a local Moonshine STT service from the ``bot_audio`` config.

    Moonshine runs on the CPU via ONNX Runtime (no GPU, no API key) and is small
    and fast. On the short, isolated bot-answer segments the harness transcribes,
    it tends to keep the answer where Whisper sometimes drops it.

    Args:
        config: The ``judge.transcription`` config mapping:

            - ``model``: Optional architecture, as a
              :class:`~pipecat.services.moonshine.stt.Model` or the equivalent
              string (default ``Model.SMALL_STREAMING``). Only ``base`` has
              non-English models, so a non-English ``language`` needs it, and
              raises at construction naming the models it does have if the
              pairing has none.
            - ``language``: Optional language code (e.g. ``es``) or ``Language``.
              When omitted, Moonshine keeps its own default (English).

    Prefer :func:`whisper_service` for a non-English bot: Moonshine's non-English
    models transcribe synthesized speech unreliably, returning an empty transcript
    or dropping the tail of an utterance, and an empty transcript is
    indistinguishable from a bot that said nothing.
    """
    from pipecat.services.moonshine.stt import Model, MoonshineSTTService

    return MoonshineSTTService(
        settings=MoonshineSTTService.Settings(
            model=config.get("model") or Model.SMALL_STREAMING,
            language=_cfg_language(config),
        ),
    )


DEFAULT_OLLAMA_JUDGE_MODEL = "gemma4:12b"

# The default judge is thinking-capable, and only its JSON verdict is ever read,
# so reasoning buys nothing while costing latency and eating into the token
# budget the verdict needs.
DEFAULT_OLLAMA_JUDGE_EXTRA = {"reasoning_effort": "none"}


def ollama_service(config: dict) -> LLMService[Any]:
    """Build a local Ollama LLM service from the ``judge:`` config.

    An ``extra:`` mapping is forwarded verbatim as top-level request parameters,
    which is how provider-specific options reach the model — notably
    ``reasoning_effort: none`` for a thinking-capable judge.

    A caller who names no model gets the default judge together with the extras
    it needs; a caller who names one gets only the extras they asked for, since
    those options are model-specific.
    """
    from pipecat.services.ollama.llm import OLLamaLLMService

    model = config.get("model")
    extra = config.get("extra")
    if extra is None:
        extra = dict(DEFAULT_OLLAMA_JUDGE_EXTRA) if model is None else {}

    base_url = config.get("endpoint") or "http://localhost:11434/v1"
    return OLLamaLLMService(
        base_url=base_url,
        settings=OLLamaLLMService.Settings(
            model=model or DEFAULT_OLLAMA_JUDGE_MODEL,
            extra=extra,
        ),
    )


def openai_service(config: dict) -> LLMService[Any]:
    """Build an OpenAI LLM service from the ``judge:`` config.

    An ``extra:`` mapping is forwarded verbatim as top-level request parameters.
    """
    from pipecat.services.openai.llm import OpenAILLMService

    return OpenAILLMService(
        settings=OpenAILLMService.Settings(
            model=config.get("model", "gpt-4o"),
            extra=config.get("extra") or {},
        )
    )
