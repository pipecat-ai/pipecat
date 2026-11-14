"""Pipecat TTS service for Faster Qwen3-TTS.

Pipecat-compatible TTS service using the ``faster-qwen3-tts`` library for
real-time speech synthesis with CUDA graph acceleration.

All configuration lives in :class:`FasterQwenTTSSettings`, which carries
working defaults for every field. Subclasses declare ``MODE`` and
``DEFAULT_MODEL``, and implement ``_build_generator``; they have no
``__init__`` of their own.

Construct via the ``settings`` keyword:

    >>> from fasterQwenTTS import FasterQwen3TTSVoiceClone, FasterQwenTTSSettings
    >>> tts = FasterQwen3TTSVoiceClone(settings=FasterQwenTTSSettings(
    ...     ref_audio="reference.wav",
    ...     ref_text="Reference transcript here.",
    ... ))

Or via the classmethod factory:

    >>> from fasterQwenTTS import FasterQwen3TTSService, FasterQwenTTSSettings
    >>> tts = FasterQwen3TTSService.create(
    ...     "voice_clone",
    ...     settings=FasterQwenTTSSettings(ref_audio="ref.wav", ref_text="tr"),
    ... )

Language is a plain string (e.g. ``"English"``, ``"Chinese"``) matching
what ``faster-qwen3-tts`` expects.
"""

from __future__ import annotations

# --- Workaround for qwen-tts-hf 0.1.1.post1 not covering the Mimi
# speech-tokenizer path for transformers 5.x. MimiConfig no longer exposes
# `rope_theta` at the top level, and ROPE_INIT_FUNCTIONS["default"] was
# removed. This restores the function that Mimi's modeling code calls.
# Remove only after the fork or upstream qwen-tts patches this path.
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

if "default" not in ROPE_INIT_FUNCTIONS:
    import torch

    def _compute_default_rope_parameters(
        config=None, device=None, seq_len=None, layer_type=None, **kwargs
    ):
        rope_params = getattr(config, "rope_parameters", None) or {}
        base = rope_params.get("rope_theta") or getattr(config, "rope_theta", 10000.0)
        partial = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial)
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
# --- End workaround ---

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, AsyncGenerator, ClassVar, Iterator, Optional

import numpy as np
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.types import NOT_GIVEN, NotGiven

try:
    from faster_qwen3_tts import FasterQwen3TTS
except ImportError as e:
    raise ImportError(
        "faster-qwen3-tts is required. Install with: pip install faster-qwen3-tts"
    ) from e


#: Sentinel returned by ``next(gen, default)`` when the generator is done.
_EXHAUSTED: object = object()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class FasterQwenTTSSettings(TTSSettings):
    """Settings for :class:`FasterQwen3TTSService`.

    Overrides the ``NOT_GIVEN`` defaults inherited from ``TTSSettings``
    with concrete values, and adds the fields specific to
    faster-qwen3-tts. ``FasterQwenTTSSettings()`` is a fully-populated,
    valid instance.
    """

    # Overridden from TTSSettings so the base class does not see NOT_GIVEN.
    model: str | None | NotGiven = None
    voice: str | None | NotGiven = None
    language: str | None | NotGiven = "English"

    # faster-qwen3-tts specific.
    ref_audio: str | None | NotGiven = None
    ref_text: str | None | NotGiven = None
    instruct: str | None | NotGiven = None
    chunk_size: int | NotGiven = 8
    device: str | NotGiven = "cuda"
    sample_rate: int | NotGiven = 24000


# ---------------------------------------------------------------------------
# Base service
# ---------------------------------------------------------------------------


class FasterQwen3TTSService(TTSService, ABC):
    """Abstract base for Faster Qwen3-TTS Pipecat services.

    Configuration is accepted through a single ``settings`` keyword. The
    base starts from ``self.Settings(model=self.DEFAULT_MODEL)`` and merges
    the caller's overrides via ``apply_update``.

    Subclasses declare ``MODE`` and ``DEFAULT_MODEL``, and implement
    ``_build_generator``.

    Do not instantiate directly. Use a concrete subclass, or
    :meth:`create` for string-keyed dispatch.
    """

    Settings = FasterQwenTTSSettings

    #: Mode name used by :meth:`create`. ``None`` on the abstract base.
    MODE: ClassVar[str | None] = None

    #: HuggingFace model id used when the caller does not set ``model``.
    DEFAULT_MODEL: ClassVar[str] = ""

    def __init__(
        self,
        *,
        settings: "FasterQwen3TTSService.Settings | None" = None,
        **kwargs,
    ):
        default_settings = self.Settings(model=self.DEFAULT_MODEL)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=default_settings.sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._model: Optional[FasterQwen3TTS] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._load_lock = asyncio.Lock()

        logger.info(
            f"{type(self).__name__} initialized: "
            f"model={default_settings.model}, "
            f"language={default_settings.language}, "
            f"sample_rate={default_settings.sample_rate}"
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        mode: str,
        *,
        settings: "FasterQwen3TTSService.Settings | None" = None,
        **kwargs,
    ) -> "FasterQwen3TTSService":
        """Construct the subclass whose ``MODE`` matches ``mode``.

        Args:
            mode: One of ``"voice_clone"``, ``"custom"``, or ``"design"``.
            settings: Overrides applied on top of the subclass defaults.
            **kwargs: Forwarded to ``TTSService.__init__``.

        Raises:
            ValueError: If ``mode`` does not match any subclass.
        """
        for sub in cls.__subclasses__():
            if sub.MODE == mode:
                return sub(settings=settings, **kwargs)
        known = sorted(
            s.MODE for s in cls.__subclasses__() if s.MODE is not None
        )
        raise ValueError(
            f"Unknown mode {mode!r}; expected one of {known}"
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pcm16(audio: np.ndarray) -> bytes:
        """Convert a float audio buffer to signed 16-bit little-endian PCM."""
        if audio.dtype != np.int16:
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767.0).astype(np.int16)
        return audio.tobytes()

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_generator(
        self, text: str
    ) -> Iterator[tuple[np.ndarray, int, Any]]:
        """Return the mode-specific streaming generator for ``text``."""
        ...

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, frame: Any) -> None:
        await super().start(frame)
        if self._model is None:
            await self._load_model()

    async def stop(self, frame: Any) -> None:
        await super().stop(frame)
        self._model = None
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        logger.info(f"{type(self).__name__} stopped, model unloaded.")

    # ------------------------------------------------------------------
    # TTSService interface
    # ------------------------------------------------------------------

    async def run_tts(
        self, text: str, context_id: str
    ) -> AsyncGenerator[Frame, None]:
        if self._model is None:
            await self._load_model()

        logger.debug(f"Generating TTS for: {text!r} (context={context_id})")

        try:
            async for audio_chunk, sr, _timing in self._stream_audio(text):
                if audio_chunk is None or len(audio_chunk) == 0:
                    continue
                yield TTSAudioRawFrame(
                    audio=self._to_pcm16(audio_chunk),
                    sample_rate=sr,
                    num_channels=1,
                )
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            yield ErrorFrame(error=str(e))

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _load_model(self) -> None:
        async with self._load_lock:
            if self._model is not None:
                return

            model_id = self._settings.model
            device = self._settings.device
            logger.info(f"Loading FasterQwen3TTS model: {model_id} ...")

            loop = asyncio.get_running_loop()
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="qwen-tts"
            )

            def _load_and_warmup() -> FasterQwen3TTS:
                model = FasterQwen3TTS.from_pretrained(model_id, device=device)
                logger.info("Warming up model and capturing CUDA graphs...")
                model.warmup(prefill_len=100)
                logger.info("CUDA graph capture complete.")
                return model

            try:
                self._model = await loop.run_in_executor(
                    self._executor, _load_and_warmup
                )
                logger.info(f"Model loaded and warmed up: {model_id}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

    async def _stream_audio(
        self, text: str
    ) -> AsyncGenerator[tuple[np.ndarray, int, Any], None]:
        gen = self._build_generator(text)
        loop = asyncio.get_running_loop()
        try:
            while True:
                item = await loop.run_in_executor(
                    self._executor, next, gen, _EXHAUSTED
                )
                if item is _EXHAUSTED:
                    return
                yield item
        finally:
            await loop.run_in_executor(self._executor, gen.close)


# ---------------------------------------------------------------------------
# Concrete modes
# ---------------------------------------------------------------------------


class FasterQwen3TTSVoiceClone(FasterQwen3TTSService):
    """Voice cloning (ICL) mode.

    Requires ``ref_audio`` and ``ref_text``:

        FasterQwen3TTSVoiceClone(settings=FasterQwenTTSSettings(
            ref_audio="speaker.wav",
            ref_text="Transcript of speaker.wav.",
        ))
    """

    MODE = "voice_clone"
    DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    def _build_generator(self, text: str) -> Iterator[tuple[np.ndarray, int, Any]]:
        return self._model.generate_voice_clone_streaming(
            text=text,
            language=self._settings.language,
            ref_audio=self._settings.ref_audio,
            ref_text=self._settings.ref_text,
            chunk_size=self._settings.chunk_size,
        )


class FasterQwen3TTSCustomVoice(FasterQwen3TTSService):
    """CustomVoice mode: a predefined speaker ID.

    Requires ``voice``:

        FasterQwen3TTSCustomVoice(settings=FasterQwenTTSSettings(voice="aiden"))
    """

    MODE = "custom"
    DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    def _build_generator(self, text: str) -> Iterator[tuple[np.ndarray, int, Any]]:
        return self._model.generate_custom_voice_streaming(
            text=text,
            language=self._settings.language,
            speaker=self._settings.voice,
            chunk_size=self._settings.chunk_size,
        )


class FasterQwen3TTSVoiceDesign(FasterQwen3TTSService):
    """VoiceDesign mode: instruction-based voice synthesis.

    Requires ``instruct``:

        FasterQwen3TTSVoiceDesign(settings=FasterQwenTTSSettings(
            instruct="Warm, confident narrator with a slight British accent",
        ))
    """

    MODE = "design"
    DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

    def _build_generator(self, text: str) -> Iterator[tuple[np.ndarray, int, Any]]:
        return self._model.generate_voice_design_streaming(
            text=text,
            language=self._settings.language,
            instruct=self._settings.instruct,
            chunk_size=self._settings.chunk_size,
        )