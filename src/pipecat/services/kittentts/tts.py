#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""KittenTTS service implementation."""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings, assert_given
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

SAMPLE_RATE = 24000
DEFAULT_MODEL = "KittenML/kitten-tts-nano-0.8"
DEFAULT_VOICE = "expr-voice-5-m"
DEFAULT_SPEED = 1.0


def _audio_to_pcm16(audio: Any) -> bytes:
    samples = np.asarray(audio, dtype=np.float32).squeeze()
    if samples.size == 0:
        return b""
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).astype("<i2").tobytes()


def _next_chunk(iterator: Any) -> Any | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


@dataclass
class KittenTTSSettings(TTSSettings):
    """Settings for KittenTTSService."""

    pass


class KittenTTSService(TTSService):
    """Local text-to-speech synthesis using KittenTTS."""

    Settings = KittenTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        model: str | None = None,
        voice_id: str | None = None,
        speed: float = DEFAULT_SPEED,
        clean_text: bool = True,
        cache_dir: str | None = None,
        backend: str | None = None,
        sample_rate: int | None = SAMPLE_RATE,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the KittenTTS service.

        Args:
            model: Hugging Face model identifier.
            voice_id: KittenTTS voice identifier.
            speed: Speech speed multiplier.
            clean_text: Whether KittenTTS should normalize text before synthesis.
            cache_dir: Optional Hugging Face cache directory.
            backend: Optional KittenTTS backend, such as ``cpu`` or ``cuda``.
            sample_rate: Output sample rate. Defaults to 24000 Hz.
            settings: Runtime-updatable settings. Values override direct model and voice args.
            **kwargs: Additional arguments passed to parent ``TTSService``.
        """
        default_settings = self.Settings(
            model=model or DEFAULT_MODEL,
            voice=voice_id or DEFAULT_VOICE,
            language=None,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._speed = speed
        self._clean_text = clean_text
        self._cache_dir = cache_dir
        self._backend = backend
        self._model: Any | None = None
        self._model_lock = asyncio.Lock()
        self._resampler = create_stream_resampler()

    def can_generate_metrics(self) -> bool:
        """Indicate that this service supports TTFB and usage metrics."""
        return True

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        changed = await super()._update_settings(delta)
        if "model" in changed:
            self._model = None
        return changed

    async def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        async with self._model_lock:
            if self._model is not None:
                return self._model

            model_id = assert_given(self._settings.model)
            if model_id is None:
                raise ValueError("KittenTTS model must be specified")

            def load_model() -> Any:
                try:
                    from kittentts import KittenTTS
                except ModuleNotFoundError as e:
                    raise ModuleNotFoundError(
                        "KittenTTS is required. Install it with "
                        "`pip install "
                        "https://github.com/KittenML/KittenTTS/releases/download/0.8.1/"
                        "kittentts-0.8.1-py3-none-any.whl`."
                    ) from e

                return KittenTTS(model_id, cache_dir=self._cache_dir, backend=self._backend)

            self._model = await asyncio.to_thread(load_model)
            return self._model

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using KittenTTS."""
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_tts_usage_metrics(text)
            model = await self._ensure_model()

            voice = assert_given(self._settings.voice)
            if voice is None:
                raise ValueError("KittenTTS voice must be specified")

            iterator = model.generate_stream(
                text,
                voice=voice,
                speed=self._speed,
                clean_text=self._clean_text,
            )

            while True:
                chunk = await asyncio.to_thread(_next_chunk, iterator)
                if chunk is None:
                    break

                await self.stop_ttfb_metrics()
                audio = _audio_to_pcm16(chunk)
                if not audio:
                    continue

                if self.sample_rate != SAMPLE_RATE:
                    audio = await self._resampler.resample(audio, SAMPLE_RATE, self.sample_rate)

                yield TTSAudioRawFrame(
                    audio=audio,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
