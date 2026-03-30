#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Typecast TTS service integration for Pipecat.

Provides text-to-speech synthesis using the Typecast API (https://typecast.ai).
Supports Korean and multilingual voices with various emotion and prosody controls.

Usage::

    from pipecat.services.typecast import TypecastTTSService

    tts = TypecastTTSService(
        api_key="YOUR_API_KEY",
        settings=TypecastTTSService.Settings(
            voice="tc_60e5426de8b95f1d3000d7b5",
            model="ssfm-v30",
        ),
    )
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
)
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts


@dataclass
class TypecastTTSSettings(TTSSettings):
    """Runtime-updatable settings for Typecast TTS service.

    Parameters:
        model: TTS model to use. Options: ``ssfm-v30`` (default), ``ssfm-v21``.
        voice: Voice ID (e.g. ``tc_60e5426de8b95f1d3000d7b5``).
        language: Language code (ISO 639-3). Auto-detected if ``None``.
        emotion_type: Emotion type. ``smart`` (ssfm-v30 only) or ``preset``.
        emotion_preset: Emotion preset. One of: ``normal``, ``happy``, ``sad``,
            ``angry``, ``whisper``, ``toneup``, ``tonedown``.
        emotion_intensity: Emotion intensity multiplier (0.0–2.0, default 1.0).
        previous_text: Context text preceding the current utterance (used with
            ``emotion_type="smart"`` for context-aware emotion).
        next_text: Context text following the current utterance.
        audio_tempo: Speech tempo multiplier (0.5–2.0, default 1.0).
        audio_pitch: Pitch adjustment in semitones (-12 to +12, default 0).
        volume: Volume level (0–200, default 100).
        seed: Random seed for reproducible output.
    """

    emotion_type: Optional[str] | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    emotion_preset: Optional[str] | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    emotion_intensity: Optional[float] | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    previous_text: Optional[str] | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    next_text: Optional[str] | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    audio_tempo: Optional[float] | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    audio_pitch: Optional[int] | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    volume: Optional[int] | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    seed: Optional[int] | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class TypecastTTSService(TTSService):
    """Typecast TTS service using the Typecast HTTP API.

    Provides text-to-speech synthesis via Typecast's REST API, supporting
    Korean and multilingual voices with emotion and prosody controls.

    The service returns WAV audio which is decoded to raw PCM for the pipeline.

    Request structure follows the official API spec::

        {
          "voice_id": "tc_...",
          "text": "...",
          "model": "ssfm-v30",
          "language": "kor",
          "prompt": {
            "emotion_type": "smart",
            "previous_text": "...",
            "next_text": "..."
          },
          "output": {
            "volume": 100,
            "audio_pitch": 0,
            "audio_tempo": 1,
            "audio_format": "wav"
          },
          "seed": 42
        }

    Example::

        tts = TypecastTTSService(
            api_key="YOUR_API_KEY",
            settings=TypecastTTSService.Settings(
                voice="tc_60e5426de8b95f1d3000d7b5",
                model="ssfm-v30",
                emotion_type="preset",
                emotion_preset="happy",
            ),
        )
    """

    Settings = TypecastTTSSettings
    _settings: Settings

    class InputParams(BaseModel):
        """Deprecated: use ``settings=TypecastTTSService.Settings(...)`` instead."""

        language: Optional[Language] = None

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: Optional[str] = None,
        model: Optional[str] = None,
        base_url: str = "https://api.typecast.ai",
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
        sample_rate: Optional[int] = None,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize the Typecast TTS service.

        Args:
            api_key: Typecast API key for authentication.
            voice_id: Voice ID to use for synthesis.

                .. deprecated::
                    Use ``settings=TypecastTTSService.Settings(voice=...)`` instead.

            model: TTS model to use (``ssfm-v30`` or ``ssfm-v21``).

                .. deprecated::
                    Use ``settings=TypecastTTSService.Settings(model=...)`` instead.

            base_url: Typecast API base URL.
            aiohttp_session: Optional shared aiohttp ClientSession. If not provided,
                one is created and managed internally.
            sample_rate: Audio sample rate. Typecast returns 44100 Hz WAV.
                Defaults to 44100.
            settings: Runtime-updatable settings object.
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        default_settings = self.Settings(
            model="ssfm-v30",
            voice=None,
            language=None,
            emotion_type=None,
            emotion_preset=None,
            emotion_intensity=None,
            previous_text=None,
            next_text=None,
            audio_tempo=None,
            audio_pitch=None,
            volume=None,
            seed=None,
        )

        if voice_id is not None:
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id

        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate or 44100,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = aiohttp_session
        self._owns_session = aiohttp_session is None

    def can_generate_metrics(self) -> bool:
        """Return True: Typecast HTTP service supports metrics generation."""
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to an ISO 639-3 language code for Typecast.

        Args:
            language: The language to convert.

        Returns:
            ISO 639-3 language code string, or None if not mapped.
        """
        _MAP = {
            Language.KO: "kor",
            Language.EN: "eng",
            Language.JA: "jpn",
            Language.ZH: "zho",
            Language.ES: "spa",
            Language.FR: "fra",
            Language.DE: "deu",
            Language.PT: "por",
            Language.RU: "rus",
            Language.IT: "ita",
            Language.NL: "nld",
            Language.PL: "pol",
            Language.TR: "tur",
            Language.VI: "vie",
            Language.ID: "ind",
        }
        return _MAP.get(language)

    async def start(self, frame: StartFrame):
        """Start the Typecast TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        if self._owns_session:
            self._session = aiohttp.ClientSession()

    async def _close_session(self):
        """Close the HTTP session if we own it."""
        if self._owns_session and self._session:
            await self._session.close()
            self._session = None

    async def stop(self, frame: EndFrame):
        """Stop the Typecast TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._close_session()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Typecast TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._close_session()

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using the Typecast HTTP API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        if not self._settings.voice:
            yield ErrorFrame(error="TypecastTTSService: no voice_id configured")
            return

        try:
            payload: dict = {
                "voice_id": self._settings.voice,
                "text": text,
                "model": self._settings.model,
            }

            if self._settings.language:
                payload["language"] = self._settings.language

            # Build prompt object (emotion control)
            prompt: dict = {}
            if self._settings.emotion_type is not None:
                prompt["emotion_type"] = self._settings.emotion_type
            if self._settings.emotion_preset is not None:
                prompt["emotion_preset"] = self._settings.emotion_preset
            if self._settings.emotion_intensity is not None:
                prompt["emotion_intensity"] = self._settings.emotion_intensity
            if self._settings.previous_text is not None:
                prompt["previous_text"] = self._settings.previous_text
            if self._settings.next_text is not None:
                prompt["next_text"] = self._settings.next_text
            if prompt:
                payload["prompt"] = prompt

            # Build output object (audio format/prosody)
            output: dict = {"audio_format": "wav"}
            if self._settings.volume is not None:
                output["volume"] = self._settings.volume
            if self._settings.audio_pitch is not None:
                output["audio_pitch"] = self._settings.audio_pitch
            if self._settings.audio_tempo is not None:
                output["audio_tempo"] = self._settings.audio_tempo
            payload["output"] = output

            if self._settings.seed is not None:
                payload["seed"] = self._settings.seed

            headers = {
                "X-API-KEY": self._api_key,
                "Content-Type": "application/json",
            }

            url = f"{self._base_url}/v1/text-to-speech"

            await self.start_ttfb_metrics()

            async with self._session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        f"{self}: Typecast API error (status {response.status}): {error_text}"
                    )
                    yield ErrorFrame(error=f"Typecast API error: {error_text}")
                    return

                wav_data = await response.read()

            await self.stop_ttfb_metrics()
            await self.start_tts_usage_metrics(text)

            # Decode WAV to raw PCM; wrap wave.Error for a clearer error message
            try:
                with wave.open(io.BytesIO(wav_data), "rb") as wf:
                    pcm_data = wf.readframes(wf.getnframes())
            except wave.Error as e:
                yield ErrorFrame(error=f"Typecast TTS: failed to decode WAV response: {e}")
                return

            frame = TTSAudioRawFrame(
                audio=pcm_data,
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=context_id,
            )

            yield frame

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
            await self.stop_ttfb_metrics()
