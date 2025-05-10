# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
from typing import AsyncGenerator, Optional

import websockets
from loguru import logger

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


class ResembleTTSService(TTSService):
    """A Text-to-Speech service using Resemble AI's streaming API.

    This service connects to Resemble AI's WebSocket API to stream TTS audio in real-time.

    Args:
        api_key: The Resemble AI API key for authentication.
        voice_uuid: The UUID of the voice to use for synthesis.
        sample_rate: The audio sample rate (default: 48000 Hz).
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice_uuid: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        **kwargs,
    ):
        """Initialize the TTS service with configuration."""
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._api_key = api_key
        self._voice_uuid = voice_uuid
        self._sample_rate = sample_rate
        self._websocket = None

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
                "no_audio_header": True
            }

            await self._websocket.send(json.dumps(request))
            await self.start_tts_usage_metrics(text)
            yield TTSStartedFrame()

            while True:
                message = await self._websocket.recv()
                data = json.loads(message)

                if data["type"] == "audio":
                    await self.stop_ttfb_metrics()
                    audio = base64.b64decode(data["audio_content"])
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
        finally:
            if self._websocket:
                await self._websocket.close()
                self._websocket = None

    async def stop(self, frame: Optional[Frame] = None):
        """Clean up resources with proper Pipecat frame handling.

        Args:
            frame: Optional frame to pass to parent stop method.
        """
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        await super().stop(frame if frame else EndFrame())