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

import aiohttp
from anam import (
    AgentAudioInputConfig,
    AnamClient,
    AnamEvent,
    ClientOptions,
    PersonaConfig,
)
from av.audio.resampler import AudioResampler
from av.video.frame import VideoFrame
from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
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

    Args:
        api_key: Anam API key for authentication.
        persona_id: ID of the persona to use (simple setup).
        persona_config: Full persona configuration (advanced setup).
        session: HTTP client session for API requests.
        ice_servers: Custom ICE servers for WebRTC (optional).\
        enable_turnkey: Whether to enable turnkey mode for Anam's all-in-one solution.
        api_base_url: Base URL for the Anam API.
        api_version: API version to use.
    """

    def __init__(
        self,
        *,
        api_key: str,
        persona_id: Optional[str] = None,
        persona_config: Optional[PersonaConfig] = None,
        session: aiohttp.ClientSession,
        ice_servers: Optional[list[dict]] = None,
        enable_turnkey: bool = False,
        api_base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the Anam video service.

        Args:
            api_key: Anam API key for authentication.
            persona_id: ID of the persona to use (simple setup).
            persona_config: Full persona configuration (advanced setup).
            session: HTTP client session for API requests.
            ice_servers: Custom ICE servers for WebRTC (optional).
            enable_turnkey: Whether to enable turnkey mode for Anam's all-in-one solution.
            api_base_url: Base URL for the Anam API.
            api_version: API version to use.
            **kwargs: Additional arguments passed to parent AIService.
        """
        super().__init__(**kwargs)
        self._api_key = api_key
        self._session = session
        self._persona_id = persona_id
        self._persona_config = persona_config
        self._ice_servers = ice_servers
        self._is_turnkey_session = enable_turnkey
        self._api_base_url = api_base_url
        self._api_version = api_version

        self._client: Optional[AnamClient] = None
        self._anam_session = None
        self._agent_audio_stream = None
        self._send_task: Optional[asyncio.Task] = None
        self._video_task: Optional[asyncio.Task] = None
        self._audio_task: Optional[asyncio.Task] = None
        self._resampler = create_stream_resampler()
        self._anam_resampler = AudioResampler("s16", "mono", 48000)
        self._is_interrupting = False
        self._transport_ready = False
        self._audio_chunk_size = 0
        self._event_id = None

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the Anam video service with necessary configuration.

        Initializes the Anam client and prepares the service for audio/video
        processing. This includes setting up audio/video streams and
        configuring callbacks.

        Args:
            setup: Configuration parameters for the frame processor.
        """
        await super().setup(setup)

        # Create persona config
        if self._persona_config:
            persona_config = self._persona_config
        elif self._persona_id:
            persona_config = PersonaConfig(persona_id=self._persona_id)
        else:
            raise ValueError("Either persona_id or persona config must be provided")

        # Create client options
        # Enable audio input for turnkey solutions (Anam handles STT)
        options = ClientOptions(
            api_base_url=self._api_base_url or "https://api.anam.ai",
            ice_servers=self._ice_servers,
            api_version=self._api_version,
        )

        # Initialize Anam client
        self._client = AnamClient(
            api_key=self._api_key,
            persona_config=persona_config,
            options=options,
        )

        # Register event handlers (only for connection events)
        self._client.add_listener(AnamEvent.CONNECTION_ESTABLISHED, self._on_connection_established)
        self._client.add_listener(AnamEvent.CONNECTION_CLOSED, self._on_connection_closed)

    async def cleanup(self):
        """Clean up the service and release resources.

        Terminates the Anam client session and cleans up associated resources.
        """
        await super().cleanup()
        await self._cancel_video_task()
        await self._cancel_audio_task()
        if self._client:
            await self._client.close()
            self._client = None
        self._anam_session = None
        self._agent_audio_stream = None

    async def start(self, frame: StartFrame):
        """Start the Anam video service and initialize the avatar session.

        Creates necessary tasks for audio/video processing and establishes
        the connection with the Anam service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if not self._client:
            raise RuntimeError("Anam client not initialized. Call setup() first.")

        # Connect to Anam
        self._anam_session = await self._client.connect_async()

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
        await self._cleanup()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Anam video service.

        Performs an immediate termination of the service, cleaning up resources
        without waiting for ongoing operations to complete.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._cleanup()

    async def _cleanup(self):
        """Clean up resources: end conversation and cancel all tasks."""
        await self._end_conversation()
        await self._cancel_video_task()
        await self._cancel_audio_task()
        await self._cancel_send_task()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and coordinate avatar behavior.

        Handles different types of frames to manage avatar interactions:
        - StartFrame: Forwards after initialization
        - EndFrame: Forwards after cleanup
        - CancelFrame: Forwards after cancellation
        - TTSAudioRawFrame: Processes audio for avatar speech (not pushed downstream)
        - InputAudioRawFrame: Processes user audio (not pushed downstream for turnkey)
        - InterruptionFrame: Handles interruptions
        - Other frames: Forwards them through the pipeline

        Args:
            frame: The frame to be processed.
            direction: The direction of frame processing (input/output).
        """
        await super().process_frame(frame, direction)

        # Handle frames that should not be pushed downstream
        if isinstance(frame, TTSAudioRawFrame):
            # Anam synchronises TTS with video frames for synchronised playback.
            await self._handle_audio_frame(frame)
            return

        if isinstance(frame, InputAudioRawFrame) and self._is_turnkey_session:
            # Anam handles STT internally, so don't push raw audio downstream for turnkey sessions.
            await self._handle_user_audio_frame(frame, direction)
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

    async def _on_connection_established(self) -> None:
        """Handle connection established event.

        Audio pushed before this point has been discarded in the SDK to limit latency build up.
        Synchronise with live point by flushing stale audio in prior buffers here, if necessary.
        """
        logger.info("Anam connection established")

    async def _on_connection_closed(self, code: str, reason: Optional[str]) -> None:
        """Handle connection closed event.

        Args:
            code: Connection close code.
            reason: Optional reason for closure.
        """
        logger.info(f"Anam connection closed: {code} - {reason or 'No reason'}")
        await self.push_error(ErrorFrame(error=f"Anam connection closed: {code}"))

    async def _handle_interruption(self):
        """Handle interruption events by resetting send tasks and notifying client.

        Manages the interruption flow by:
        1. Setting the interruption flag
        2. Signaling the session to interrupt current speech
        3. Cancelling ongoing audio sending tasks
        4. Creating a new send task
        """
        self._is_interrupting = True
        if self._anam_session:
            self._anam_session.interrupt()
        await self._cancel_send_task()
        self._is_interrupting = False
        await self._create_send_task()

    async def _end_conversation(self):
        """End the current conversation and reset state.

        Closes the Anam session and cleans up conversation-specific resources.
        """
        if self._anam_session:
            await self._anam_session.close()
            self._anam_session = None
        self._agent_audio_stream = None

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
        """Queue an audio frame for processing.

        Places the audio frame in the processing queue for synchronized
        delivery to the Anam service.

        Args:
            frame: The audio frame to process.
        """
        await self._queue.put(frame)

    async def _handle_user_audio_frame(self, frame: InputAudioRawFrame, direction: FrameDirection):
        """Handle user audio frame by sending it to Anam SDK.

        For turnkey solutions where Anam handles STT, this sends user audio
        directly to the SDK via send_user_audio(). The SDK handles WebRTC
        transmission and format conversion internally.

        Args:
            frame: The user audio frame to process (InputAudioRawFrame).
            direction: The direction of frame processing.
        """
        if self._client is None or self._client._streaming_client is None:
            return

        try:
            # Send raw audio samples to SDK for WebRTC transport to Anam's service
            self._client._streaming_client.send_user_audio(
                audio_bytes=frame.audio,
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
            )
        except Exception as e:
            logger.error(f"Failed to send user audio to Anam SDK: {e}")
            await self.push_error(ErrorFrame(error=f"Failed to send user audio: {e}"))

    async def _send_task_handler(self):
        """Handle sending audio frames to the Anam client.

        Continuously processes audio frames from the queue and sends them to the
        Anam client. Handles timeouts and silence detection for proper audio
        streaming management.

        Anam works best with 16 bit PCM 24kHz mono audio.
        Audio send to Anam is returned in-sync with the avatar without any resampling.
        Sample rates lower than 24kHz will result in poor Avatar performance.
        Sample rates highet than 24kHz might impact latency without any noticeable improvement in audio quality.
        """
        if not self._agent_audio_stream:
            logger.error("Agent audio stream not initialized")
            return

        audio_buffer = bytearray()
        self._event_id = None

        while True:
            try:
                frame = await asyncio.wait_for(self._queue.get(), timeout=AVATAR_VAD_STOP_SECS)
                if self._is_interrupting:
                    break

                if isinstance(frame, TTSAudioRawFrame):
                    # Starting a new inference
                    if self._event_id is None:
                        self._event_id = str(frame.id)

                    # 500ms of audio: sample_rate * 0.5 seconds * 2 bytes per sample (16-bit)
                    self._audio_chunk_size = int(frame.sample_rate * 0.5 * 2)

                    audio_buffer.extend(frame.audio)

                    # Send chunks when we have enough data
                    while len(audio_buffer) >= self._audio_chunk_size:
                        chunk = bytes(audio_buffer[: self._audio_chunk_size])
                        audio_buffer = audio_buffer[self._audio_chunk_size :]
                        await self._agent_audio_stream.send_audio_chunk(chunk)

                self._queue.task_done()
            except asyncio.TimeoutError:
                # Bot has stopped speaking - signal end of sequence
                if self._event_id is not None and self._agent_audio_stream:
                    await self._agent_audio_stream.end_sequence()
                    self._event_id = None
                    audio_buffer.clear()
            except Exception as e:
                logger.error(f"Error in audio send task: {e}")
                await self.push_error(ErrorFrame(error=f"Anam audio send error: {e}"))
                break
