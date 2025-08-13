# SPDX-License-Identifier: BSD-2-Clause
"""Vonage WebSocket transport (chunk iterator + sleep-per-chunk pacing)."""

from __future__ import annotations

import asyncio
import io
import wave
from typing import Optional

from loguru import logger

from pipecat.frames.frames import Frame, OutputAudioRawFrame
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.network.websocket_server import (
    WebsocketServerOutputTransport,
    WebsocketServerParams,
    WebsocketServerTransport,
)

# ---- Constants ---------------------------------------------------------------

DEFAULT_WS_HOST: str = "localhost"
DEFAULT_WS_PORT: int = 8765
PCM16_SAMPLE_WIDTH_BYTES: int = 2  # 16-bit PCM


class VonageWebsocketServerTransport(WebsocketServerTransport):
    """WebSocket server transport that paces by sleeping once per audio chunk."""

    def __init__(
        self,
        params: WebsocketServerParams,
        host: str = DEFAULT_WS_HOST,
        port: int = DEFAULT_WS_PORT,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ) -> None:
        """Initialize the Vonage WebSocket server transport.

        Args:
            params: WebSocket server parameters including serializer and audio options.
            host: Host address for the WebSocket server.
            port: Port number for the WebSocket server.
            input_name: Optional name for the input transport.
            output_name: Optional name for the output transport.
        """
        super().__init__(params, host, port, input_name, output_name)
        self._params = params

    def output(self) -> WebsocketServerOutputTransport:
        """Return the output transport used to send data to clients."""
        if not self._output:
            self._output = VonageWebsocketServerOutputTransport(self, self._params)
        return self._output


class VonageWebsocketServerOutputTransport(WebsocketServerOutputTransport):
    """Output transport that sends each serializer-produced chunk and sleeps between sends."""

    def __init__(self, transport: BaseTransport, params: WebsocketServerParams, **kwargs) -> None:
        """Initialize the Vonage WebSocket output transport.

        Args:
            transport: The base transport instance to wrap.
            params: WebSocket server parameters.
            **kwargs: Additional keyword arguments for the base class.
        """
        super().__init__(transport, params, **kwargs)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> None:
        """Write an audio frame to the WebSocket client with pacing."""
        if not self._websocket:
            # Keep pipeline timing consistent if the client isn't connected yet.
            await self._write_audio_sleep()
            return

        normalized = OutputAudioRawFrame(
            audio=frame.audio,
            sample_rate=self.sample_rate,
            num_channels=self._params.audio_out_channels,
        )

        if self._params.add_wav_header:
            with io.BytesIO() as buffer:
                with wave.open(buffer, "wb") as wf:
                    wf.setsampwidth(PCM16_SAMPLE_WIDTH_BYTES)
                    wf.setnchannels(normalized.num_channels)
                    wf.setframerate(normalized.sample_rate)
                    wf.writeframes(normalized.audio)
                normalized = OutputAudioRawFrame(
                    audio=buffer.getvalue(),
                    sample_rate=normalized.sample_rate,
                    num_channels=normalized.num_channels,
                )

        await self._write_frame(normalized)

    async def _write_frame(self, frame: Frame) -> None:
        """Serialize and send a frame to the WebSocket client."""
        if not self._params.serializer:
            return

        try:
            payload = await self._params.serializer.serialize(frame)
            if payload and self._websocket:
                # For audio, serializer returns a list[bytes] of chunks.
                # Pace by sleeping once per chunk using serializer's interval.
                for chunk in payload:
                    await self._websocket.send(chunk)
                    await asyncio.sleep(self._params.serializer.sleep_interval)
        except Exception as exc:
            logger.error(f"{self} exception sending data: {exc.__class__.__name__} ({exc})")
