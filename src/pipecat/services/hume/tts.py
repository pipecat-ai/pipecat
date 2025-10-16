# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License

"""Hume Text-to-Speech service implementation."""

import base64
import os
from typing import Any, AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from hume import AsyncHumeClient
    from hume.tts import (
        FormatPcm,
        PostedUtterance,
        PostedUtteranceVoiceWithId,
    )
except ModuleNotFoundError as e:  # pragma: no cover - import-time guidance
    logger.error(f"Exception: {e}")
    logger.error("In order to use Hume, you need to `pip install pipecat-ai[hume]`.")
    raise Exception(f"Missing module: {e}")


HUME_SAMPLE_RATE = 48_000  # Hume TTS streams at 48 kHz


class HumeTTSService(TTSService):
    """Hume Octave Text-to-Speech service.

    Streams PCM audio via Hume's HTTP output streaming (JSON chunks) endpoint
    using the Python SDK and emits ``TTSAudioRawFrame`` frames suitable for Pipecat transports.

    Supported features:

    - Generates speech from text using Hume TTS.
    - Streams PCM audio.
    - Supports dynamic updates of voice and synthesis parameters at runtime.
    - Provides metrics for Time To First Byte (TTFB) and TTS usage.
    """

    class InputParams(BaseModel):
        """Optional synthesis parameters for Hume TTS.

        Parameters:
            description: Natural-language acting directions (up to 100 characters).
            speed: Speaking-rate multiplier (0.5-2.0).
            trailing_silence: Seconds of silence to append at the end (0-5).
        """

        description: Optional[str] = None
        speed: Optional[float] = None
        trailing_silence: Optional[float] = None

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        voice_id: str,
        params: Optional[InputParams] = None,
        sample_rate: Optional[int] = HUME_SAMPLE_RATE,
        **kwargs,
    ) -> None:
        """Initialize the HumeTTSService.

        Args:
            api_key: Hume API key. If omitted, reads the ``HUME_API_KEY`` environment variable.
            voice_id: ID of the voice to use. Only voice IDs are supported; voice names are not.
            params: Optional synthesis controls (acting instructions, speed, trailing silence).
            sample_rate: Output sample rate for emitted PCM frames. Defaults to 48_000 (Hume).
            **kwargs: Additional arguments passed to the parent class.
        """
        api_key = api_key or os.getenv("HUME_API_KEY")
        if not api_key:
            raise ValueError("HumeTTSService requires an API key (env HUME_API_KEY or api_key=)")

        if sample_rate != HUME_SAMPLE_RATE:
            logger.warning(
                f"Hume TTS streams at {HUME_SAMPLE_RATE} Hz; configured sample_rate={sample_rate}"
            )

        super().__init__(sample_rate=sample_rate, **kwargs)

        self._client = AsyncHumeClient(api_key=api_key)
        self._params = params or HumeTTSService.InputParams()

        # Store voice in the base class (mirrors other services)
        self.set_voice(voice_id)

        self._audio_bytes = b""

    def can_generate_metrics(self) -> bool:
        """Can generate metrics.

        Returns:
            True if metrics can be generated, False otherwise.
        """
        return True

    async def start(self, frame: StartFrame) -> None:
        """Start the service.

        Args:
            frame: The start frame.
        """
        await super().start(frame)

    async def update_setting(self, key: str, value: Any) -> None:
        """Runtime updates via `TTSUpdateSettingsFrame`.

        Args:
            key: The name of the setting to update. Recognized keys are:
                - "voice_id"
                - "description"
                - "speed"
                - "trailing_silence"
            value: The new value for the setting.
        """
        key_l = (key or "").lower()

        if key_l == "voice_id":
            self.set_voice(str(value))
            logger.info(f"HumeTTSService voice_id set to: {self.voice}")
        elif key_l == "description":
            self._params.description = None if value is None else str(value)
        elif key_l == "speed":
            self._params.speed = None if value is None else float(value)
        elif key_l == "trailing_silence":
            self._params.trailing_silence = None if value is None else float(value)
        else:
            # Defer unknown keys to the base class
            await super().update_setting(key, value)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Hume TTS.

        Args:
            text: The text to be synthesized.

        Returns:
            An async generator that yields `Frame` objects, including
            `TTSStartedFrame`, `TTSAudioRawFrame`, `ErrorFrame`, and
            `TTSStoppedFrame`.
        """
        logger.debug(f"{self}: Generating Hume TTS: [{text}]")

        # Build the request payload
        utterance_kwargs: dict[str, Any] = {
            "text": text,
            "voice": PostedUtteranceVoiceWithId(id=self._voice_id),
        }
        if self._params.description is not None:
            utterance_kwargs["description"] = self._params.description
        if self._params.speed is not None:
            utterance_kwargs["speed"] = self._params.speed
        if self._params.trailing_silence is not None:
            utterance_kwargs["trailing_silence"] = self._params.trailing_silence

        utterance = PostedUtterance(**utterance_kwargs)

        # Request raw PCM chunks in the streaming JSON
        pcm_fmt = FormatPcm(type="pcm")

        await self.start_ttfb_metrics()
        await self.start_tts_usage_metrics(text)
        yield TTSStartedFrame()

        try:
            # Instant mode is always enabled here (not user-configurable)
            # Hume emits mono PCM at 48 kHz; downstream can resample if needed.
            # We buffer audio bytes before sending to prevent glitches.
            self._audio_bytes = b""
            async for chunk in self._client.tts.synthesize_json_streaming(
                utterances=[utterance],
                format=pcm_fmt,
                instant_mode=True,
                version="2",
            ):
                audio_b64 = getattr(chunk, "audio", None)
                if not audio_b64:
                    continue

                pcm_bytes = base64.b64decode(audio_b64)
                self._audio_bytes += pcm_bytes

                # Buffer audio until we have enough to avoid glitches
                if len(self._audio_bytes) < self.chunk_size:
                    continue

                frame = TTSAudioRawFrame(
                    audio=self._audio_bytes,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )

                yield frame

                self._audio_bytes = b""

        except Exception as e:
            logger.exception(f"{self} error generating TTS: {e}")
            await self.push_error(ErrorFrame(f"Error generating TTS: {e}"))
        finally:
            # Ensure TTFB timer is stopped even on early failures
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
