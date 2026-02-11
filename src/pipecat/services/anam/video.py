#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Anam video service implementation for Pipecat.

This module provides integration with Anam.ai for creating conversational
AI applications with avatars. It manages conversation sessions and provides
real-time audio/video streaming capabilities through the Anam API.
"""

import asyncio
from typing import Optional

from anam import (
    AgentAudioInputConfig,
    AnamClient,
    AnamEvent,
    ClientOptions,
    ConnectionClosedCode,
    PersonaConfig,
)
from av.audio.resampler import AudioResampler
from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    OutputImageRawFrame,
    OutputTransportReadyFrame,
    SpeechOutputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.ai_service import AIService

# Using the same values that we do in the BaseOutputTransport
AVATAR_VAD_STOP_SECS = 0.35


class AnamVideoService(AIService):
    """A service that integrates Anam.ai's avatar capabilities into the pipeline.

    This service manages the lifecycle of an Anam avatar session by handling
    bidirectional audio/video streaming, avatar animations, and user interactions.
    It processes various frame types to coordinate the avatar's behavior and
    maintains synchronization between audio and video streams.

    The service supports:

    - Real-time avatar animation based on audio input
    - Voice activity detection for natural interactions
    - Interrupt handling for more natural conversations
    - Audio resampling for optimal playback quality
    - Automatic session management
    """

    def __init__(
        self,
        *,
        api_key: str,
        persona_config: PersonaConfig,
        ice_servers: Optional[list[dict]] = None,
        api_base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the Anam video service.

        Args:
            api_key: Anam API key for authentication.
            persona_config: Full persona configuration.
            ice_servers: Custom ICE servers for WebRTC (optional).
            api_base_url: Base URL for the Anam API.
            api_version: API version to use.
            **kwargs: Additional arguments passed to parent AIService.
        """
        super().__init__(**kwargs)
        self._api_key = api_key
        self._persona_config = persona_config
        self._ice_servers = ice_servers
        self._api_base_url = api_base_url
        self._api_version = api_version

        self._client: Optional[AnamClient] = None
        self._anam_session = None
        self._agent_audio_stream = None
        self._send_task: Optional[asyncio.Task] = None
        self._video_task: Optional[asyncio.Task] = None
        self._audio_task: Optional[asyncio.Task] = None
        self._anam_resampler = AudioResampler("s16", "mono", 48000)
        self._is_interrupting = False
        self._transport_ready = False
        self._session_ready_event = asyncio.Event()

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the Anam video service with necessary configuration.

        Initializes the Anam client and prepares the service for audio/video
        processing. This includes setting up audio/video streams and
        configuring callbacks.

        Args:
            setup: Configuration parameters for the frame processor.
        """
        await super().setup(setup)

        # Create client options
        options = ClientOptions(
            api_base_url=self._api_base_url or "https://api.anam.ai",
            ice_servers=self._ice_servers,
            api_version=self._api_version,
        )

        # Initialize Anam client
        self._client = AnamClient(
            api_key=self._api_key,
            persona_config=self._persona_config,
            options=options,
        )

        # Register event handlers (only for connection events)
        self._client.add_listener(AnamEvent.SESSION_READY, self._on_session_ready)
        self._client.add_listener(AnamEvent.CONNECTION_CLOSED, self._on_connection_closed)

    async def cleanup(self):
        """Clean up the service and release resources.

        Terminates the Anam client session and cleans up associated resources.
        """
        await super().cleanup()
        await self._close_session()
        await self._cleanup()

    async def start(self, frame: StartFrame):
        """Start the Anam video service and initialize the avatar session.

        Creates necessary tasks for audio/video processing and establishes
        the connection with the Anam service. Blocks until sessionready is
        received so that audio is only forwarded when the backend is ready.

        Args:
            frame: The start frame containing initialization parameters.
        """
        if not self._client:
            raise RuntimeError("Anam client not initialized. Call setup() first.")

        self._session_ready_event.clear()

        self._anam_session = await self._client.connect_async()

        # Block until sessionready so the backend is ready to receive TTS
        await self._session_ready_event.wait()

        # Allow the pipeline to continue start up
        await super().start(frame)

        # Create agent audio input stream for sending TTS audio
        audio_config = AgentAudioInputConfig(
            encoding="pcm_s16le",
            sample_rate=24000,
            channels=1,
        )
        self._agent_audio_stream = self._anam_session.create_agent_audio_input_stream(audio_config)

        # Create tasks for consuming video and audio frames
        self._video_task = self.create_task(self._consume_video_frames())
        self._audio_task = self.create_task(self._consume_audio_frames())

        # Create send task
        await self._create_send_task()

    async def stop(self, frame: EndFrame):
        """Stop the Anam video service gracefully.

        Performs cleanup by ending the conversation and cancelling ongoing tasks
        in a controlled manner.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._close_session()
        await self._cleanup()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Anam video service.

        Performs an immediate termination of the service, cleaning up resources
        without waiting for ongoing operations to complete.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._close_session()
        await self._cleanup()

    async def _cleanup(self):
        """Clean up resources: end conversation and cancel all tasks."""
        await self._cancel_video_task()
        await self._cancel_audio_task()
        await self._cancel_send_task()
        self._agent_audio_stream = None
        self._is_interrupting = False
        self._transport_ready = False
        self._client = None
        self._anam_session = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and coordinate avatar behavior.

        Handles different types of frames to manage avatar interactions:

        - TTSAudioRawFrame: Processes audio for avatar speech (not pushed downstream)
        - InterruptionFrame: Handles interruptions
        - OutputTransportReadyFrame: Sets transport ready flag
        - TTSStartedFrame: Starts TTFB metrics
        - BotStartedSpeakingFrame: Stops TTFB metrics
        - Other frames: Forwards them through the pipeline

        Args:
            frame: The frame to be processed.
            direction: The direction of frame processing (input/output).
        """
        await super().process_frame(frame, direction)

        # Handle frames that should not be pushed downstream
        if isinstance(frame, TTSAudioRawFrame):
            # Do not forward TTS audio downstream as Anam synchronises TTS with video frames for synchronised playback.
            await self._handle_audio_frame(frame)
            return

        # Handle frames that need processing before being pushed downstream
        if isinstance(frame, InterruptionFrame):
            await self._handle_interruption()
        if isinstance(frame, OutputTransportReadyFrame):
            self._transport_ready = True
        if isinstance(frame, TTSStartedFrame):
            await self.start_ttfb_metrics()
        if isinstance(frame, BotStartedSpeakingFrame):
            await self.stop_ttfb_metrics()

        # Push frames downstream
        await self.push_frame(frame, direction)

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True if metrics generation is supported.
        """
        return True

    async def _consume_video_frames(self) -> None:
        """Consume video frames from Anam iterator and push them downstream."""
        if not self._anam_session:
            return

        try:
            async for video_frame in self._anam_session.video_frames():
                if not self._transport_ready:
                    continue

                frame = OutputImageRawFrame(
                    image=video_frame.to_ndarray(format="rgb24").tobytes(),
                    size=(video_frame.width, video_frame.height),
                    format="RGB",
                )

                await self.push_frame(frame)
        except Exception as e:
            logger.error(f"Error consuming video frames: {e}")
            await self.push_error(ErrorFrame(error=f"Anam video frame error: {e}"))

    async def _consume_audio_frames(self) -> None:
        """Consume audio frames from Anam iterator and push them downstream."""
        if not self._anam_session:
            return

        try:
            async for audio_frame in self._anam_session.audio_frames():
                if not self._transport_ready:
                    continue

                resampled_audio = self._anam_resampler.resample(audio_frame)
                for resampled_frame in resampled_audio:
                    frame = SpeechOutputAudioRawFrame(
                        audio=resampled_frame.to_ndarray().tobytes(),
                        sample_rate=self._anam_resampler.rate,
                        num_channels=self._anam_resampler.layout.nb_channels,
                    )
                    await self.push_frame(frame)
        except Exception as e:
            logger.error(f"Error consuming audio frames: {e}")
            await self.push_error(ErrorFrame(error=f"Anam audio frame error: {e}"))

    async def _cancel_video_task(self):
        """Cancel the video frame consumption task if it exists."""
        if self._video_task:
            await self.cancel_task(self._video_task)
            self._video_task = None

    async def _cancel_audio_task(self):
        """Cancel the audio frame consumption task if it exists."""
        if self._audio_task:
            await self.cancel_task(self._audio_task)
            self._audio_task = None

    async def _on_session_ready(self) -> None:
        """Handle session ready event (backend service is ready to receive audio).

        Unblocks the pipeline so StartFrame can propagate and audio can be forwarded to the backend.
        Any audio pushed before this point has been discarded and lost.
        """
        logger.info("Anam connection established")
        self._session_ready_event.set()

    async def _on_connection_closed(self, code: str, reason: Optional[str]) -> None:
        """Handle connection closed event.

        Args:
            code: Connection close code (from ConnectionClosedCode enum).
            reason: Optional reason for closure.
        """
        if code != ConnectionClosedCode.NORMAL.value:
            error_message = f"Anam connection closed: {code}"
            if reason:
                error_message += f" - {reason}"
            logger.error(f"{error_message}")
            await self._cleanup()
            await self.push_error(ErrorFrame(error=error_message))

    async def _handle_interruption(self) -> None:
        """Handle interruption events by resetting send tasks and notifying client.

        Manages the interruption flow by:
        1. Setting the interruption flag
        2. Signaling the session to interrupt current speech and signal end_sequence
        3. Cancelling ongoing audio sending tasks
        4. Creating a new send task
        """
        self._is_interrupting = True
        if self._anam_session:
            await self._anam_session.interrupt()
            if self._agent_audio_stream:
                await self._agent_audio_stream.end_sequence()

        await self._cancel_send_task()
        self._is_interrupting = False
        await self._create_send_task()

    async def _close_session(self):
        """Close the Anam client."""
        if self._client and self._anam_session and self._anam_session.is_active:
            logger.debug("Disconnecting from Anam")
            await self._anam_session.close()
            self._anam_session = None
            self._client = None

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

    async def _handle_audio_frame(self, frame: TTSAudioRawFrame):
        """Queue an audio frame for sending to Anam."""
        await self._queue.put(frame)

    async def _send_task_handler(self):
        """Handle sending audio frames to the Anam client.

        Sends each TTS frame's audio directly to the backend without buffering.
        Sends end_sequence when queue is empty.

        Anam works best with 16 bit PCM 24kHz mono audio.
        """
        if not self._agent_audio_stream:
            logger.error("Agent audio stream not initialized")
            return

        while True:
            try:
                frame = await asyncio.wait_for(self._queue.get(), timeout=AVATAR_VAD_STOP_SECS)
                if self._is_interrupting:
                    break

                if isinstance(frame, TTSAudioRawFrame) and frame.audio:
                    await self._agent_audio_stream.send_audio_chunk(frame.audio)

            except asyncio.TimeoutError:
                if self._agent_audio_stream:
                    await self._agent_audio_stream.end_sequence()
            except Exception as e:
                logger.error(f"Error in audio send task: {e}")
                await self.push_error(ErrorFrame(error=f"Anam audio send error: {e}"))
                break
