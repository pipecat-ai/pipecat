# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
import pyaudio
from typing import AsyncGenerator, Optional

import websockets
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    ErrorFrame,
    EndFrame,
)
from pipecat.services.tts_service import TTSService


# Audio configuration constants
DEFAULT_SAMPLE_RATE = 48000
CHUNK_SIZE = 1024
AUDIO_FORMAT = pyaudio.paInt16


class ResembleTTSService(TTSService):
    """A Text-to-Speech service using Resemble AI's streaming API.

    This service connects to Resemble AI's WebSocket API to stream TTS audio in real-time.
    It supports configurable audio parameters like sample rate, speed, and pitch.

    Args:
        api_key: The Resemble AI API key for authentication.
        voice_uuid: The UUID of the voice to use for synthesis.
        sample_rate: The audio sample rate (default: 48000 Hz).
        params: Optional parameters for speech synthesis (speed, pitch).
    """

    class InputParams(BaseModel):
        """Optional parameters for speech synthesis.

        Attributes:
            speed: Optional speed adjustment (1.0 is normal speed).
            pitch: Optional pitch adjustment (1.0 is normal pitch).
        """
        speed: Optional[float] = None
        pitch: Optional[float] = None

    def __init__(
        self,
        *,
        api_key: str,
        voice_uuid: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        """Initialize the TTS service with configuration."""
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._api_key = api_key
        self._voice_uuid = voice_uuid
        self._params = params
        self._websocket = None
        self._sample_rate = sample_rate

        # Initialize PyAudio
        self._pyaudio = pyaudio.PyAudio()
        self._stream = self._pyaudio.open(
            format=AUDIO_FORMAT,
            channels=1,
            rate=self._sample_rate,
            output=True,
            frames_per_buffer=CHUNK_SIZE,
        )

    def can_generate_metrics(self) -> bool:
        """Whether this service can generate TTS metrics."""
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate TTS audio for the given text.

        Args:
            text: The text to synthesize.

        Yields:
            Frame objects representing the TTS process (start, audio, stop, or error).
        """
        logger.debug(f"Generating TTS for: {text}")

        try:
            # Connect to WebSocket
            self._websocket = await websockets.connect(
                "wss://websocket.cluster.resemble.ai/stream",
                extra_headers={"Authorization": f"Bearer {self._api_key}"},
                ping_interval=5,
                ping_timeout=20,
            )

            await self.start_ttfb_metrics()

            # Prepare request
            request = {
                "voice_uuid": self._voice_uuid,
                "data": text,
                "sample_rate": self._sample_rate,
                "precision": "PCM_16",
            }

            # Add optional parameters
            if self._params.speed is not None:
                request["speed"] = self._params.speed
            if self._params.pitch is not None:
                request["pitch"] = self._params.pitch

            await self._websocket.send(json.dumps(request))
            await self.start_tts_usage_metrics(text)
            yield TTSStartedFrame()

            while True:
                message = await self._websocket.recv()
                data = json.loads(message)

                if data["type"] == "audio":
                    await self.stop_ttfb_metrics()
                    audio = base64.b64decode(data["audio_content"])
                    self._stream.write(audio)  # Play audio immediately
                    yield TTSAudioRawFrame(audio, self._sample_rate, 1)

                elif data["type"] == "audio_end":
                    yield TTSStoppedFrame()
                    break

                elif data["type"] == "error":
                    error_msg = data.get("message", "Unknown error")
                    yield ErrorFrame(error=f"API Error: {error_msg}")
                    break

        except websockets.ConnectionClosed:
            yield ErrorFrame(error="Connection closed unexpectedly")
        except Exception as e:
            logger.error(f"Error during TTS: {e}")
            yield ErrorFrame(error=str(e))

    async def stop(self, frame: Optional[Frame] = None):
        """Clean up resources with proper Pipecat frame handling.

        Args:
            frame: Optional frame to pass to parent stop method.
        """
        if self._websocket:
            await self._websocket.close()
        if hasattr(self, "_stream"):
            self._stream.stop_stream()
            self._stream.close()
        if hasattr(self, "_pyaudio"):
            self._pyaudio.terminate()
        # Call parent with EndFrame if none provided
        await super().stop(frame if frame else EndFrame())