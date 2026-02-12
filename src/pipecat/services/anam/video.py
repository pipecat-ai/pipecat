#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Anam video service implementation for Pipecat.

This module provides integration with Anam.ai for creating interactive avatars
through Anam's Python SDK. It uses audio input and provides realistic avatars
as synchronized raw audio/video frames.
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

# Time between TTS frames to signal end_sequence. Makes avatar return to listening mode.
TTS_TIMEOUT = 0.2  # seconds


class AnamVideoService(AIService):
    """Anam.ai's Video service that generates real-time interactive avatars from audio.

    This service uses Anam's Python SDK to manage sessions and communication with Anam's backend.
    It consumes audio and user interactions and receives synchronized audio/video frames. The SDK
    provides decoded WebRTC audio and video frames as PyAV objects. Ingested audio is passed through
    without resampling, but has been resampled to 48kHz stereo for webRTC delivery to the SDK.


    The service supports:

    - Real-time avatar animation based on audio input
    - Voice activity detection for natural interactions
    - Interrupt handling for more natural conversations
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
        self._transport_ready = False
        self._session_ready_event = asyncio.Event()

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the Anam video service with necessary configuration.

        Initializes the Anam client and prepares the service for audio/video
        processing. Sets up audio/video streams and registers event handlers.

        Args:
            setup: Configuration parameters for the frame processor.
        """
        await super().setup(setup)

        # Initialize Anam client
        self._client = AnamClient(
            api_key=self._api_key,
            persona_config=self._persona_config,
            options=ClientOptions(
                api_base_url=self._api_base_url or "https://api.anam.ai",
                ice_servers=self._ice_servers,
                api_version=self._api_version,
            ),
        )

        # Register event handlers
        self._client.add_listener(AnamEvent.SESSION_READY, self._on_session_ready)
        self._client.add_listener(AnamEvent.CONNECTION_CLOSED, self._on_connection_closed)

    async def cleanup(self):
        """Clean up the service and release resources."""
        await super().cleanup()
        await self._close_session()
        await self._cleanup()

    async def start(self, frame: StartFrame):
        """Start the Anam video service and initialize the avatar session.

        Creates an Anam session and creates tasks to forward audio/video. Blocks until
        session_ready is received so audio is not dropped before backend is ready to receive audio.

        Args:
            frame: The start frame containing initialization parameters.
        """
        if not self._client:
            raise RuntimeError("Anam client not initialized. Call setup() first.")

        self._session_ready_event.clear()

        try:
            self._anam_session = await self._client.connect_async()
        except Exception as e:
            logger.error(f"Error connecting to Anam: {e}")
            await self.push_error(ErrorFrame(error=f"Anam connection error: {e}"))
            raise

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
        try:
            self._agent_audio_stream = self._anam_session.create_agent_audio_input_stream(
                audio_config
            )
        except Exception as e:
            logger.error(f"Error creating agent audio stream: {e}")
            await self.push_error(ErrorFrame(error=f"Anam agent audio stream error: {e}"))
            raise

        # Create tasks for consuming video and audio frames
        self._video_task = self.create_task(self._consume_video_frames())
        self._audio_task = self.create_task(self._consume_audio_frames())

        # Create send task
        await self._create_send_task()

    async def stop(self, frame: EndFrame):
        """Stop the Anam video service gracefully.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._close_session()
        await self._cleanup()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Anam video service.

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

        if isinstance(frame, TTSAudioRawFrame):
            # Do not forward TTS audio downstream; Anam syncs TTS with video
            await self._queue.put(frame)
            return

        if isinstance(frame, InterruptionFrame):
            await self._handle_interruption()
        if isinstance(frame, OutputTransportReadyFrame):
            self._transport_ready = True
        if isinstance(frame, TTSStartedFrame):
            await self.start_ttfb_metrics()
        if isinstance(frame, BotStartedSpeakingFrame):
            await self.stop_ttfb_metrics()

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
        """Consume audio frames from Anam iterator and push them downstream.

        Audio frames are decoded WebRTC OPUS: 16 bit 48kHz stereo PCM samples.
        """
        if not self._anam_session:
            return

        try:
            async for audio_frame in self._anam_session.audio_frames():
                if not self._transport_ready:
                    continue

                # Resample to mono as some downstream transports cannot handle stereo audio.
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

        Unblocks the pipeline to propagate StartFrame and allow audio to be ingested.
        """
        logger.info("Anam connection established")
        self._session_ready_event.set()

    async def _on_connection_closed(self, code: str, reason: Optional[str]) -> None:
        """Handle connection closed event.

        Client and session are closed by the SDK prior to emitting this event.

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
        1. Signaling the session to interrupt current speech
        2. Cancelling the send task (stops sending; discards old queue)
        3. Calling end_sequence to set avatar in listening mode
        4. Creating a new send task with a fresh queue
        """
        if self._anam_session:
            await self._anam_session.interrupt()

        await self._cancel_send_task()
        if self._agent_audio_stream:
            await self._agent_audio_stream.end_sequence()
        await self._create_send_task()

    async def _close_session(self):
        """Close the Anam client."""
        if self._client and self._anam_session and self._anam_session.is_active:
            try:
                logger.debug("Disconnecting from Anam")
                await self._anam_session.close()
            except Exception as e:
                logger.warning(f"Error closing Anam session: {e}")
            finally:
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

    async def _send_task_handler(self):
        """Handle sending audio frames to the Anam client.

        Forwards each TTS frame's audio without buffering. Requires end_sequence after last
        audio frame. Using timeout as TTSStoppedFrame can arrive too soon.

        Anam accepts any sample rate but recommends 16 bit PCM 24kHz mono audio for best trade-off
        between latency, audio quality and avatar performance.
        """
        if not self._agent_audio_stream:
            logger.error("Agent audio stream not initialized")
            return

        while True:
            try:
                frame = await asyncio.wait_for(self._queue.get(), timeout=TTS_TIMEOUT)
                if isinstance(frame, TTSAudioRawFrame) and frame.audio:
                    await self._agent_audio_stream.send_audio_chunk(frame.audio)

            except asyncio.TimeoutError:
                if self._agent_audio_stream:
                    await self._agent_audio_stream.end_sequence()
            except Exception as e:
                logger.error(f"Error in audio send task: {e}")
                await self.push_error(ErrorFrame(error=f"Anam audio send error: {e}"))
                break
