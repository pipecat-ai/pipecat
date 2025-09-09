#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""HeyGen implementation for Pipecat.

This module provides integration with the HeyGen platform for creating conversational
AI applications with avatars. It manages conversation sessions and provides real-time
audio/video streaming capabilities through the HeyGen API.
"""

import asyncio
from typing import Optional

import aiohttp
from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AudioRawFrame,
    BotStartedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    ImageRawFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    OutputTransportReadyFrame,
    SpeechOutputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.ai_service import AIService
from pipecat.services.heygen.api import NewSessionRequest
from pipecat.services.heygen.client import HEY_GEN_SAMPLE_RATE, HeyGenCallbacks, HeyGenClient
from pipecat.transports.base_transport import TransportParams

# Using the same values that we do in the BaseOutputTransport
AVATAR_VAD_STOP_SECS = 0.35


class HeyGenVideoService(AIService):
    """A service that integrates HeyGen's interactive avatar capabilities into the pipeline.

    This service manages the lifecycle of a HeyGen avatar session by handling bidirectional
    audio/video streaming, avatar animations, and user interactions. It processes various frame types
    to coordinate the avatar's behavior and maintains synchronization between audio and video streams.

    The service supports:

    - Real-time avatar animation based on audio input
    - Voice activity detection for natural interactions
    - Interrupt handling for more natural conversations
    - Audio resampling for optimal quality
    - Automatic session management

    Args:
        api_key (str): HeyGen API key for authentication
        session (aiohttp.ClientSession): HTTP client session for API requests
        session_request (NewSessionRequest, optional): Configuration for the HeyGen session.
            Defaults to using the "Shawn_Therapist_public" avatar with "v2" version.
    """

    def __init__(
        self,
        *,
        api_key: str,
        session: aiohttp.ClientSession,
        session_request: NewSessionRequest = NewSessionRequest(avatar_id="Shawn_Therapist_public"),
        **kwargs,
    ) -> None:
        """Initialize the HeyGen video service.

        Args:
            api_key: HeyGen API key for authentication
            session: HTTP client session for API requests
            session_request: Configuration for the HeyGen session (default: uses Shawn_Therapist_public avatar)
            **kwargs: Additional arguments passed to parent AIService
        """
        super().__init__(**kwargs)
        self._api_key = api_key
        self._session = session
        self._client: Optional[HeyGenClient] = None
        self._send_task: Optional[asyncio.Task] = None
        self._resampler = create_stream_resampler()
        self._is_interrupting = False
        self._session_request = session_request
        self._other_participant_has_joined = False
        self._event_id = None
        self._audio_chunk_size = 0

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the HeyGen video service with necessary configuration.

        Initializes the HeyGen client, establishes connections, and prepares the service
        for audio/video processing. This includes setting up audio/video streams,
        configuring callbacks, and initializing the resampler.

        Args:
            setup: Configuration parameters for the frame processor.
        """
        await super().setup(setup)
        self._client = HeyGenClient(
            api_key=self._api_key,
            session=self._session,
            params=TransportParams(
                audio_in_enabled=True,
                video_in_enabled=True,
                audio_out_enabled=True,
                audio_out_sample_rate=HEY_GEN_SAMPLE_RATE,
            ),
            session_request=self._session_request,
            callbacks=HeyGenCallbacks(
                on_participant_connected=self._on_participant_connected,
                on_participant_disconnected=self._on_participant_disconnected,
            ),
        )
        await self._client.setup(setup)

    async def cleanup(self):
        """Clean up the service and release resources.

        Terminates the HeyGen client session and cleans up associated resources.
        """
        await super().cleanup()
        await self._client.cleanup()
        self._client = None

    async def _on_participant_connected(self, participant_id: str):
        """Handle participant connected events."""
        logger.info(f"Participant connected {participant_id}")
        if not self._other_participant_has_joined:
            self._other_participant_has_joined = True
            await self._client.capture_participant_video(
                participant_id, self._on_participant_video_frame
            )
            await self._client.capture_participant_audio(
                participant_id, self._on_participant_audio_data
            )

    async def _on_participant_disconnected(self, participant_id: str):
        """Handle participant disconnected events."""
        logger.info(f"Participant disconnected {participant_id}")

    async def _on_participant_video_frame(self, video_frame: ImageRawFrame):
        """Handle incoming video frames from participants."""
        frame = OutputImageRawFrame(
            image=video_frame.image,
            size=video_frame.size,
            format=video_frame.format,
        )
        await self.push_frame(frame)

    async def _on_participant_audio_data(self, audio_frame: AudioRawFrame):
        """Handle incoming audio data from participants."""
        frame = SpeechOutputAudioRawFrame(
            audio=audio_frame.audio,
            sample_rate=audio_frame.sample_rate,
            num_channels=audio_frame.num_channels,
        )
        await self.push_frame(frame)

    async def start(self, frame: StartFrame):
        """Start the HeyGen video service and initialize the avatar session.

        Creates necessary tasks for audio/video processing and establishes
        the connection with the HeyGen service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        # 40 ms of audio, match the default behavior from the output transport
        self._audio_chunk_size = int((HEY_GEN_SAMPLE_RATE * 2) / 25)
        await self._client.start(frame, self._audio_chunk_size)
        await self._create_send_task()

    async def stop(self, frame: EndFrame):
        """Stop the HeyGen video service gracefully.

        Performs cleanup by ending the conversation and cancelling ongoing tasks
        in a controlled manner.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._end_conversation()
        await self._cancel_send_task()

    async def cancel(self, frame: CancelFrame):
        """Cancel the HeyGen video service.

        Performs an immediate termination of the service, cleaning up resources
        without waiting for ongoing operations to complete.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._end_conversation()
        await self._cancel_send_task()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and coordinate avatar behavior.

        Handles different types of frames to manage avatar interactions:
        - UserStartedSpeakingFrame: Activates avatar's listening animation
        - UserStoppedSpeakingFrame: Deactivates avatar's listening state
        - TTSAudioRawFrame: Processes audio for avatar speech
        - Other frames: Forwards them through the pipeline

        Args:
            frame: The frame to be processed.
            direction: The direction of frame processing (input/output).
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            await self._handle_user_started_speaking()
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._client.stop_agent_listening()
            await self.push_frame(frame, direction)
        elif isinstance(frame, OutputTransportReadyFrame):
            self._client.transport_ready()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSAudioRawFrame):
            await self._handle_audio_frame(frame)
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

    async def _handle_user_started_speaking(self):
        """Handle the event when a user starts speaking.

        Manages the interruption flow by:
        1. Setting the interruption flag
        2. Signaling the client to interrupt current speech
        3. Cancelling ongoing audio sending tasks
        4. Creating a new send task
        5. Activating the avatar's listening animation
        """
        self._is_interrupting = True
        await self._client.interrupt(self._event_id)
        await self._cancel_send_task()
        self._is_interrupting = False
        await self._create_send_task()
        await self._client.start_agent_listening()

    async def _end_conversation(self):
        """End the current conversation and reset state.

        Stops the HeyGen client and cleans up conversation-specific resources.
        """
        self._other_participant_has_joined = False
        await self._client.stop()

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
        """Queue an audio frame for processing.

        Places the audio frame in the processing queue for synchronized
        delivery to the HeyGen service.

        Args:
            frame: The audio frame to process.
        """
        await self._queue.put(frame)

    async def _send_task_handler(self):
        """Handle sending audio frames to the HeyGen client.

        Continuously processes audio frames from the queue and sends them to the
        HeyGen client. Handles timeouts and silence detection for proper audio
        streaming management.
        """
        sample_rate = self._client.out_sample_rate
        audio_buffer = bytearray()
        self._event_id = None

        while True:
            try:
                frame = await asyncio.wait_for(self._queue.get(), timeout=AVATAR_VAD_STOP_SECS)
                if self._is_interrupting:
                    break
                if isinstance(frame, TTSAudioRawFrame):
                    # starting the new inference
                    if self._event_id is None:
                        self._event_id = str(frame.id)

                    audio = await self._resampler.resample(
                        frame.audio, frame.sample_rate, sample_rate
                    )
                    audio_buffer.extend(audio)
                    while len(audio_buffer) >= self._audio_chunk_size:
                        chunk = audio_buffer[: self._audio_chunk_size]
                        audio_buffer = audio_buffer[self._audio_chunk_size :]

                        await self._client.agent_speak(bytes(chunk), self._event_id)
                self._queue.task_done()
            except asyncio.TimeoutError:
                # Bot has stopped speaking
                if self._event_id is not None:
                    await self._client.agent_speak_end(self._event_id)
                    self._event_id = None
                    audio_buffer.clear()
