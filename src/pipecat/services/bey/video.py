#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Beyond Presence implementation for Pipecat.

This module provides integration with the Beyond Presence API to generate avatar videos
starting from voice agents.
"""

import asyncio

import aiohttp

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    Frame,
    SpeechOutputAudioRawFrame,
    StartFrame,
    StartInterruptionFrame,
    TransportMessageFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService
from pipecat.transports.services.daily import DailyTransportClient

BASE_API_URL = "https://api.bey.dev/v1"
FRAME_RATE = 25


class BeyVideoService(AIService):
    """A service that integrates Beyond Presence's avatar video generation into the pipeline.

    Converts audio stream from the pipeline into an avatar video stream posted directly
    to a Daily room from an external worker managed by Beyond Presence.
    """

    def __init__(
        self,
        api_key: str,
        avatar_id: str,
        transport_client: DailyTransportClient,
        session: aiohttp.ClientSession,
        **kwargs,
    ) -> None:
        """Initialize the Beyond Presence speech-to-video service.

        Args:
            api_key: Beyond Presence API key used for authentication.
            avatar_id: ID of the Beyond Presence avatar to use for video synthesis.
            transport_client: DailyTransportClient for managing WebRTC connections.
            session: Async HTTP session used for communication with Beyond Presence.
            **kwargs: Additional arguments passed to the parent AIService class.
        """
        super().__init__(**kwargs)

        self._api_key = api_key
        self._avatar_id = avatar_id
        self._transport_client = transport_client
        self._session = session

        self._resampler = create_stream_resampler()
        self._queue = asyncio.Queue()
        self._out_sample_rate = 24000
        self._audio_buffer = bytearray()
        self._transport_destination: str = "bey-custom-track"
        self._http_session: aiohttp.ClientSession | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames through the service.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._start_session(
                room_url=self._transport_client.room_url,
                token=self._transport_client._token,
            )
            await self._transport_client.register_audio_destination(self._transport_destination)
            await self.push_frame(frame, direction)
        elif isinstance(frame, StartInterruptionFrame):
            frame.transport_destination = self._transport_destination
            transport_frame = TransportMessageFrame(message="interrupt")
            await self._transport_client.send_message(transport_frame)
        elif isinstance(frame, TTSAudioRawFrame):
            in_sample_rate = frame.sample_rate
            chunk_size = int((self._out_sample_rate * 2) / FRAME_RATE)

            resampled = await self._resampler.resample(
                frame.audio, in_sample_rate, self._out_sample_rate
            )
            self._audio_buffer.extend(resampled)
            while len(self._audio_buffer) >= chunk_size:
                chunk = SpeechOutputAudioRawFrame(
                    bytes(self._audio_buffer[:chunk_size]),
                    sample_rate=self._out_sample_rate,
                    num_channels=frame.num_channels,
                )

                chunk.transport_destination = self._transport_destination

                self._audio_buffer = self._audio_buffer[chunk_size:]
                await self._transport_client.write_audio_frame(chunk)
        elif isinstance(frame, TTSStartedFrame):
            await self.start_ttfb_metrics()
        elif isinstance(frame, BotStartedSpeakingFrame):
            # We constantly receive audio through WebRTC, but most of the time it is silence.
            # As soon as we receive actual audio, the base output transport will create a
            # BotStartedSpeakingFrame, which we can use as a signal for the TTFB metrics.
            await self.stop_ttfb_metrics()
        else:
            await self.push_frame(frame, direction)

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True if metrics generation is supported.
        """
        return True

    async def _start_session(self, room_url: str, token: str) -> None:
        async with self._session().post(
            f"{BASE_API_URL}/session",
            headers={
                "x-api-key": self._api_key,
            },
            json={
                "avatar_id": self._avatar_id,
                "transport_type": "pipecat",
                # TODO: we might want to rename these to just url and token
                "pipecat_url": room_url,
                "pipecat_token": token,
            },
        ) as response:
            if not response.ok:
                text = await response.text()
                raise Exception("Server returned an error", status_code=response.status, body=text)
            return
