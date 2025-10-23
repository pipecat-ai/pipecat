#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tavus video service implementation for avatar-based video generation.

This module implements Tavus as a sink transport layer, providing video
avatar functionality through Tavus's streaming API.
"""

import asyncio
from typing import Optional

import aiohttp
from daily.daily import AudioData, VideoFrame
from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    OutputTransportReadyFrame,
    SpeechOutputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.ai_service import AIService
from pipecat.transports.tavus.transport import TavusCallbacks, TavusParams, TavusTransportClient


class TavusVideoService(AIService):
    """Service that proxies audio to Tavus and receives audio and video in return.

    Uses the TavusTransportClient to manage sessions and handle communication.
    When audio is sent, Tavus responds with both audio and video streams, which
    are routed through Pipecat's media pipeline.

    In use cases with DailyTransport, this creates two distinct virtual rooms:

    - Tavus room: Contains the Tavus Avatar and the Pipecat Bot
    - User room: Contains the Pipecat Bot and the user
    """

    def __init__(
        self,
        *,
        api_key: str,
        replica_id: str,
        persona_id: str = "pipecat-stream",
        session: aiohttp.ClientSession,
        **kwargs,
    ) -> None:
        """Initialize the Tavus video service.

        Args:
            api_key: Tavus API key used for authentication.
            replica_id: ID of the Tavus voice replica to use for speech synthesis.
            persona_id: ID of the Tavus persona. Defaults to "pipecat-stream" for Pipecat TTS voice.
            session: Async HTTP session used for communication with Tavus.
            **kwargs: Additional arguments passed to the parent AIService class.
        """
        super().__init__(**kwargs)
        self._api_key = api_key
        self._session = session
        self._replica_id = replica_id
        self._persona_id = persona_id

        self._other_participant_has_joined = False
        self._client: Optional[TavusTransportClient] = None

        self._conversation_id: str
        self._resampler = create_stream_resampler()

        self._audio_buffer = bytearray()
        self._send_task: Optional[asyncio.Task] = None
        # This is the custom track destination expected by Tavus
        self._transport_destination: Optional[str] = "stream"
        self._transport_ready = False

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the Tavus video service.

        Args:
            setup: Frame processor setup configuration.
        """
        await super().setup(setup)
        callbacks = TavusCallbacks(
            on_participant_joined=self._on_participant_joined,
            on_participant_left=self._on_participant_left,
        )
        self._client = TavusTransportClient(
            bot_name="Pipecat",
            callbacks=callbacks,
            api_key=self._api_key,
            replica_id=self._replica_id,
            persona_id=self._persona_id,
            session=self._session,
            params=TavusParams(
                audio_in_enabled=True,
                video_in_enabled=True,
                audio_out_enabled=True,
                microphone_out_enabled=False,
            ),
        )
        await self._client.setup(setup)

    async def cleanup(self):
        """Clean up the service and release resources."""
        await super().cleanup()
        await self._client.cleanup()
        self._client = None

    async def _on_participant_left(self, participant, reason):
        """Handle participant leaving the session."""
        participant_id = participant["id"]
        logger.info(f"Participant left {participant_id}, reason: {reason}")

    async def _on_participant_joined(self, participant):
        """Handle participant joining the session."""
        participant_id = participant["id"]
        logger.info(f"Participant joined {participant_id}")
        if not self._other_participant_has_joined:
            self._other_participant_has_joined = True
            await self._client.capture_participant_video(
                participant_id, self._on_participant_video_frame, 30
            )
            await self._client.capture_participant_audio(
                participant_id=participant_id,
                callback=self._on_participant_audio_data,
                sample_rate=self._client.out_sample_rate,
            )

    async def _on_participant_video_frame(
        self, participant_id: str, video_frame: VideoFrame, video_source: str
    ):
        """Handle incoming video frames from participants."""
        frame = OutputImageRawFrame(
            image=video_frame.buffer,
            size=(video_frame.width, video_frame.height),
            format=video_frame.color_format,
        )
        frame.transport_source = video_source
        if self._transport_ready:
            await self.push_frame(frame)

    async def _on_participant_audio_data(
        self, participant_id: str, audio: AudioData, audio_source: str
    ):
        """Handle incoming audio data from participants."""
        frame = SpeechOutputAudioRawFrame(
            audio=audio.audio_frames,
            sample_rate=audio.sample_rate,
            num_channels=audio.num_channels,
        )
        frame.transport_source = audio_source
        if self._transport_ready:
            await self.push_frame(frame)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Tavus service supports metrics generation.
        """
        return True

    async def get_persona_name(self) -> str:
        """Get the name of the current persona.

        Returns:
            The persona name from the Tavus client.
        """
        return await self._client.get_persona_name()

    async def start(self, frame: StartFrame):
        """Start the Tavus video service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._client.start(frame)
        if self._transport_destination:
            await self._client.register_audio_destination(self._transport_destination)
        await self._create_send_task()

    async def stop(self, frame: EndFrame):
        """Stop the Tavus video service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._end_conversation()
        await self._cancel_send_task()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Tavus video service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._end_conversation()
        await self._cancel_send_task()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames through the service.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            await self._handle_interruptions()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSAudioRawFrame):
            await self._handle_audio_frame(frame)
        elif isinstance(frame, OutputTransportReadyFrame):
            self._transport_ready = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSStartedFrame):
            await self.start_ttfb_metrics()
        elif isinstance(frame, BotStartedSpeakingFrame):
            # We constantly receive audio through WebRTC, but most of the time it is silence.
            # As soon as we receive actual audio, the base output transport will create a
            # BotStartedSpeakingFrame, which we can use as a signal for the TTFB metrics.
            await self.stop_ttfb_metrics()
        else:
            await self.push_frame(frame, direction)

    async def _handle_interruptions(self):
        """Handle interruption events by resetting send tasks and notifying client."""
        await self._cancel_send_task()
        await self._create_send_task()
        await self._client.send_interrupt_message()

    async def _end_conversation(self):
        """End the current conversation and reset state."""
        await self._client.stop()
        self._other_participant_has_joined = False

    async def _create_send_task(self):
        """Create the audio sending task if it doesn't exist."""
        if not self._send_task:
            self._queue = asyncio.Queue()
            self._send_task = self.create_task(self._send_task_handler())

    async def _cancel_send_task(self):
        """Cancel the audio sending task if it exists."""
        if self._send_task:
            await self.cancel_task(self._send_task)
            self._send_task = None

    async def _handle_audio_frame(self, frame: OutputAudioRawFrame):
        """Process audio frames for sending to Tavus."""
        sample_rate = self._client.out_sample_rate
        # 40 ms of audio
        chunk_size = int((sample_rate * 2) / 25)
        # We might need to resample if incoming audio doesn't match the
        # transport sample rate.
        resampled = await self._resampler.resample(frame.audio, frame.sample_rate, sample_rate)
        self._audio_buffer.extend(resampled)
        while len(self._audio_buffer) >= chunk_size:
            chunk = OutputAudioRawFrame(
                bytes(self._audio_buffer[:chunk_size]),
                sample_rate=sample_rate,
                num_channels=frame.num_channels,
            )
            chunk.transport_destination = self._transport_destination
            await self._queue.put(chunk)
            self._audio_buffer = self._audio_buffer[chunk_size:]

    async def _send_task_handler(self):
        """Handle sending audio frames to the Tavus client."""
        while True:
            frame = await self._queue.get()
            if isinstance(frame, OutputAudioRawFrame) and self._client:
                await self._client.write_audio_frame(frame)
            self._queue.task_done()
