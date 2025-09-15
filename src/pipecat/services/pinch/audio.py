#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pinch implementation for Pipecat.

This module provides integration with the Pinch platform for real-time translation
with audio streaming. It manages translation sessions and provides real-time
audio streaming capabilities through the Pinch API.
"""

import asyncio
from typing import Optional

import aiohttp
from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    OutputTransportReadyFrame,
    SpeechOutputAudioRawFrame,
    StartFrame,
    TranscriptionFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.ai_service import AIService
from pipecat.transports.pinch.api import (
    PinchSessionRequest,
    PinchConnectionError,
    PinchConfigurationError,
)
from pipecat.transports.pinch.client import PINCH_OUTPUT_SAMPLE_RATE, PinchCallbacks, PinchClient
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.time import time_now_iso8601


class PinchAudioService(AIService):
    """A service that integrates Pinch's translation capabilities into the pipeline.

    This service manages the lifecycle of a Pinch translation session by handling bidirectional
    audio streaming and transcript processing. It processes various frame types
    to coordinate translation behavior and maintains synchronization between audio and transcript streams.

    The service supports:

    - Real-time audio translation via LiveKit audio tracks
    - Voice activity detection for natural interactions
    - Transcript processing for both original and translated text
    - Audio resampling for optimal quality
    - Automatic session management
    - Direct audio streaming over LiveKit for enhanced reliability

    Args:
        api_token (str): Pinch API token for authentication
        session (aiohttp.ClientSession): HTTP client session for API requests
        session_request (PinchSessionRequest, optional): Configuration for the Pinch session.
            Defaults to English to Spanish translation with female voice.
    """

    def __init__(
        self,
        *,
        api_token: str,
        session: aiohttp.ClientSession,
        session_request: PinchSessionRequest = PinchSessionRequest(),
        **kwargs,
    ) -> None:
        """Initialize the Pinch audio service.

        Args:
            api_token: Pinch API token for authentication
            session: HTTP client session for API requests
            session_request: Configuration for the Pinch session
            **kwargs: Additional arguments passed to parent AIService
        """
        super().__init__(**kwargs)
        self._api_token = api_token
        self._session = session
        self._client: Optional[PinchClient] = None
        self._resampler = create_stream_resampler()
        self._session_request = session_request
        self._session_active = False
        # Backpressure and pacing
        self._audio_queue: Optional[asyncio.Queue] = None
        self._audio_task: Optional[asyncio.Task] = None
        self._max_queue_size = 10  # Limit queue size to prevent unbounded growth

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the Pinch audio service with necessary configuration.

        Initializes the Pinch client, establishes connections, and prepares the service
        for audio/transcript processing. This includes setting up audio streams,
        configuring callbacks, and initializing the resampler.

        Args:
            setup: Configuration parameters for the frame processor.
        """
        await super().setup(setup)

        # Initialize audio queue for backpressure
        self._audio_queue = asyncio.Queue(maxsize=self._max_queue_size)

        self._client = PinchClient(
            api_token=self._api_token,
            session=self._session,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_out_sample_rate=PINCH_OUTPUT_SAMPLE_RATE,
            ),
            session_request=self._session_request,
            callbacks=PinchCallbacks(
                on_original_transcript=self._on_original_transcript,
                on_translated_transcript=self._on_translated_transcript,
                on_audio_data=self._on_audio_data,
                on_session_started=self._on_session_started,
                on_session_ended=self._on_session_ended,
            ),
        )
        await self._client.setup(setup)

        # Start audio processing task
        self._audio_task = setup.task_manager.create_task(
            self._process_audio_queue(),
            f"{self}::audio_queue_processor"
        )

    async def cleanup(self):
        """Clean up the service and release resources.

        Terminates the Pinch client session and cleans up associated resources.
        """
        await super().cleanup()

        # Cancel audio processing task
        if self._audio_task and not self._audio_task.done():
            self._audio_task.cancel()
            try:
                await self._audio_task
            except asyncio.CancelledError:
                pass
            self._audio_task = None

        if self._client:
            await self._client.cleanup()
            self._client = None

    async def _on_original_transcript(self, text: str, is_final: bool):
        """Handle original transcript from Pinch."""
        if is_final:
            logger.info(f"🎤 Original transcript: {text}")
            await self.push_frame(
                TranscriptionFrame(
                    text=text,
                    user_id="user",
                    timestamp=time_now_iso8601(),
                ),
                FrameDirection.UPSTREAM,
            )
        else:
            await self.push_frame(
                InterimTranscriptionFrame(
                    text=text,
                    user_id="user",
                    timestamp=time_now_iso8601(),
                ),
                FrameDirection.UPSTREAM,
            )

    async def _on_translated_transcript(self, text: str, is_final: bool):
        """Handle translated transcript from Pinch."""
        if is_final:
            logger.info(f"🎯 Translation: {text}")
            # Push as LLMTextFrame and TTSTextFrame
            # This allows the text to be captured by context aggregators, rtvi observers, etc.
            await self.push_frame(LLMTextFrame(text=text))
            await self.push_frame(TTSTextFrame(text=text))

    async def _on_audio_data(self, audio: bytes):
        """Handle translated audio data from Pinch."""
        await self.push_frame(
            SpeechOutputAudioRawFrame(
                audio=audio,
                sample_rate=PINCH_OUTPUT_SAMPLE_RATE,
                num_channels=1,
            ),
            FrameDirection.DOWNSTREAM,
        )

    async def _on_session_started(self):
        """Handle session started event."""
        self._session_active = True
        logger.info("Pinch session started")

    async def _on_session_ended(self):
        """Handle session ended event."""
        self._session_active = False
        logger.info("Pinch session ended")
        # End any ongoing response
        # RTVI Observer will automatically detect these and send bot-stopped-speaking
        await self.push_frame(TTSStoppedFrame())
        await self.push_frame(LLMFullResponseEndFrame())

    async def start(self, frame: StartFrame):
        """Start the Pinch audio service and initialize the translation session.

        Creates necessary tasks for audio/transcript processing and establishes
        the connection with the Pinch service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        # Use standard audio chunk size
        await self._client.start(frame)


    async def stop(self, frame: EndFrame):
        """Stop the Pinch audio service gracefully.

        Performs cleanup by ending the conversation and cancelling ongoing tasks
        in a controlled manner.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._end_conversation()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Pinch audio service.

        Performs an immediate termination of the service, cleaning up resources
        without waiting for ongoing operations to complete.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._end_conversation()

    async def _process_audio_queue(self):
        """Process audio frames from the queue with pacing."""
        while True:
            try:
                # Get audio frame from queue
                audio_data, sample_rate = await self._audio_queue.get()

                # Send to Pinch client
                if self._client:
                    await self._client.send_audio(audio_data, sample_rate)

                self._audio_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing audio from queue: {e}")
                self._audio_queue.task_done()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and coordinate translation behavior.

        Handles different types of frames to manage translation interactions:
        - InputAudioRawFrame: Queues audio for translation via LiveKit audio tracks
        - UserStartedSpeakingFrame: Signals start of user speech
        - UserStoppedSpeakingFrame: Signals end of user speech
        - Other frames: Forwards them through the pipeline

        Args:
            frame: The frame to be processed.
            direction: The direction of frame processing (input/output).
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            # Queue audio for processing with backpressure
            try:
                await asyncio.wait_for(
                    self._audio_queue.put((frame.audio, frame.sample_rate)),
                    timeout=0.1  # Short timeout to avoid blocking
                )
            except asyncio.TimeoutError:
                logger.warning("Audio queue full, dropping frame to prevent latency buildup")
            except asyncio.QueueFull:
                logger.warning("Audio queue full, dropping frame to prevent latency buildup")

            await self.push_frame(frame, direction)

        elif isinstance(frame, UserStartedSpeakingFrame):
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self.push_frame(frame, direction)
        elif isinstance(frame, OutputTransportReadyFrame):
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True if metrics generation is supported.
        """
        return False

    async def _end_conversation(self):
        """End the current conversation and reset state.

        Stops the Pinch client and cleans up conversation-specific resources.
        """
        self._session_active = False
        if self._client:
            await self._client.stop()
