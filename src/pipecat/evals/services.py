#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Service constructors for the eval harness: a Pipecat service from a scenario's config mapping.

The dispatchers pick a provider by its ``service:`` name, or call a
``factory``; the per-provider builders do the rest. Provider imports stay
inside the functions, so importing this module is cheap.
"""

import importlib
import os
from typing import TYPE_CHECKING, Any

from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.types import NOT_GIVEN, NotGiven

if TYPE_CHECKING:
    from pipecat.evals.tts import CachingTTSService


def _cfg_language(cfg: dict) -> Language | NotGiven:
    """A config's optional ``language`` as a :class:`Language`; a code string is accepted too.

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


def stt_service_from_config(config: dict | None) -> STTService:
    """Build the STT that transcribes the bot's audio, from a ``judge.transcription:`` mapping.

    A ``factory`` (a dotted path to a callable taking the config) builds it;
    otherwise the ``service`` name picks one, ``moonshine`` by default. Any
    pipeline STT works; the pipeline sets its sample rate.

    Args:
        config: The ``transcription`` mapping, or ``None`` for the Moonshine default.

    Returns:
        A constructed ``STTService`` (model loaded), ready to add to the pipeline.
    """
    config = config or {}

    custom = config.get("factory")
    if custom:
        module_name, _, attr = custom.rpartition(".")
        if not module_name:
            raise ValueError(f"transcription.factory must be a dotted path: {custom!r}")
        factory = getattr(importlib.import_module(module_name), attr)
        return factory(config)

    name = str(config.get("service", "moonshine")).lower()
    if name == "whisper":
        return whisper_service(config)
    if name == "moonshine":
        return moonshine_service(config)

    raise ValueError(
        f"Unknown STT service: {name!r}. Known: whisper, moonshine. "
        "Or set transcription.factory to a 'module.func' returning an STTService."
    )


def tts_service_from_config(
    voice_cfg: dict,
    *,
    cache_dir: str | None = None,
    use_cache: bool = True,
) -> "CachingTTSService":
    """Build the user-audio TTS, wrapped in a cache, from a ``user_audio`` mapping.

    A ``factory`` (a dotted path to a callable taking the voice config) builds
    the inner service; otherwise the ``service`` name picks one, ``kokoro``
    or ``cartesia``. The wrapper synthesizes each user utterance once and
    reuses it across runs. The pipeline sets the sample rate.

    Args:
        voice_cfg: ``user_audio`` mapping — ``service`` and ``voice`` at minimum;
            optional ``model`` / ``api_key`` for Cartesia.
        cache_dir: Where to store cached audio (see ``CachingTTSService``).
        use_cache: When False, force fresh synthesis.

    Returns:
        A configured ``CachingTTSService`` (not yet started).
    """
    # Lazy import to keep this module cheap and avoid importing the TTS stack
    # unless a scenario actually needs synthesized user audio.
    from pipecat.evals.tts import CachingTTSService, tts_cache_key

    custom = voice_cfg.get("factory")
    if custom:
        module_name, _, attr = custom.rpartition(".")
        if not module_name:
            raise ValueError(f"user_audio.factory must be a dotted path: {custom!r}")
        factory = getattr(importlib.import_module(module_name), attr)
        inner = factory(voice_cfg)
    else:
        name = str(voice_cfg.get("service", "")).lower()
        voice = str(voice_cfg.get("voice", ""))
        if not name or not voice:
            raise ValueError("user_audio config requires at least 'service' and 'voice'")
        if name == "kokoro":
            inner = kokoro_service(voice_cfg)
        elif name == "cartesia":
            inner = cartesia_service(voice_cfg)
        else:
            raise ValueError(
                f"Unknown TTS service: {name!r}. Known: kokoro, cartesia. "
                "Or set user_audio.factory to a 'module.func' returning a TTSService."
            )

    return CachingTTSService(
        inner,
        cache_key=tts_cache_key(voice_cfg),
        cache_dir=cache_dir,
        use_cache=use_cache,
    )


def kokoro_service(voice_cfg: dict) -> TTSService:
    """Build a local Kokoro TTS service from the ``user_audio`` config.

    No API key and no per-run cost; the model is downloaded once and cached
    under ``~/.cache/pipecat/kokoro-onnx``.

    Args:
        voice_cfg: The ``user.speech`` config mapping:

            - ``voice``: Kokoro voice id (e.g. ``af_heart``).
            - ``language``: Optional language code (e.g. ``zh``) or ``Language``.
              When omitted, Kokoro keeps its own default (English). Voices are
              language-specific, so a non-English language needs a matching voice
              — ``af_heart`` speaks US English whatever the language is set to.
    """
    from pipecat.services.kokoro.tts import KokoroTTSService

    return KokoroTTSService(
        settings=KokoroTTSService.Settings(
            voice=str(voice_cfg.get("voice", "")),
            language=_cfg_language(voice_cfg),
        ),
    )


def cartesia_service(voice_cfg: dict) -> TTSService:
    """Build a Cartesia TTS service from the ``user_audio`` config.

    Args:
        voice_cfg: The ``user.speech`` config mapping:

            - ``voice``: Cartesia voice id.
            - ``model``: Optional model (defaults to ``sonic-2``).
            - ``api_key``: Optional key (falls back to ``$CARTESIA_API_KEY``).
            - ``language``: Optional language code (e.g. ``zh``) or ``Language``.
              When omitted, Cartesia keeps its own default (English).

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
    )


def whisper_service(config: dict) -> STTService:
    """Build a local Whisper STT service from the ``bot_audio`` config.

    It runs on the CPU by default, leaving the GPU to the judge and the audio
    models; transcription is off the hot path, so the latency is fine. Set
    ``device: cuda`` (and ``compute_type``) in the ``transcription`` config
    for GPU.

    Whisper's non-speech filter is disabled: it only ever hears the bot's own
    speech, and the default threshold drops correct transcriptions of TTS
    speech run to run.

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

    Small and fast on the CPU, and steadier than Whisper on the short
    bot-answer segments the harness transcribes.

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


def llm_service_from_config(config: dict | None, *, where: str) -> LLMService[Any]:
    """Build an LLM service from a ``service:`` block, the judge's or a persona's.

    A ``factory`` (a dotted path to a callable taking the config) builds it;
    otherwise the ``service`` name picks a provider, ``ollama`` by default.

    Args:
        config: Mapping with keys ``service`` (default ``"ollama"``), ``model``,
            optional ``endpoint``, and an optional ``extra`` mapping forwarded to
            the model as top-level request parameters. ``None`` uses all defaults.
        where: The config block's name in the file, for error messages
            (``"judge.eval"``, ``"simulator"``).

    Returns:
        The configured LLM service.

    Raises:
        ValueError: If ``service`` is unknown or ``factory`` is not a dotted path.
    """
    config = config or {}
    custom = config.get("factory")
    if custom:
        module_name, _, attr = custom.rpartition(".")
        if not module_name:
            raise ValueError(f"{where}.factory must be a dotted path: {custom!r}")
        factory = getattr(importlib.import_module(module_name), attr)
        return factory(config)
    service_name = str(config.get("service", "ollama")).lower()
    if service_name == "ollama":
        return ollama_service(config)
    if service_name == "openai":
        return openai_service(config)
    raise ValueError(
        f"Unknown {where} service: {service_name!r}. Known: ollama, openai. "
        f"Or set {where}.factory to a 'module.func' returning an LLM service."
    )


def ollama_service(config: dict) -> LLMService[Any]:
    """Build a local Ollama LLM service from the ``judge:`` config.

    An ``extra:`` mapping goes to the model as request parameters, which is
    how ``reasoning_effort: none`` reaches a thinking model. The default
    model comes with the extras it needs; a named model gets only the extras
    asked for.
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
