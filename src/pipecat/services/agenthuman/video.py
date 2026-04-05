#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AgentHuman implementation for Pipecat.

This module provides integration with the AgentHuman platform for creating conversational
AI applications with avatars. It manages conversation sessions and provides real-time
audio/video streaming capabilities through the AgentHuman API.
"""

from typing import Optional

from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.transports.base_transport import BaseTransport
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
    UserStartedSpeakingFrame
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.ai_service import AIService
from .api import NewSessionRequest
from .client import AGENTHUMAN_SAMPLE_RATE, AgentHumanCallbacks, AgentHumanClient
from pipecat.transports.base_transport import TransportParams

_SUPPORTED_ASPECT_RATIOS: dict[str, float] = {
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "1:1": 1.0,
}
_ASPECT_RATIO_TOLERANCE = 0.02


def _nearest_aspect_ratio(width: int, height: int) -> str:
    """Return the name of the supported aspect ratio closest to width/height."""
    actual = width / height
    return min(_SUPPORTED_ASPECT_RATIOS, key=lambda r: abs(_SUPPORTED_ASPECT_RATIOS[r] - actual))


def _is_standard_ratio(width: int, height: int) -> bool:
    """Return True if width/height is within tolerance of any supported ratio."""
    actual = width / height
    return any(
        abs(actual - v) <= _ASPECT_RATIO_TOLERANCE
        for v in _SUPPORTED_ASPECT_RATIOS.values()
    )


class AgentHumanVideoService(AIService):
    """A service that integrates AgentHuman's interactive avatar capabilities into the pipeline.

    This service manages the lifecycle of a AgentHuman avatar session by handling bidirectional
    audio/video streaming, avatar animations, and user interactions. It processes various frame types
    to coordinate the avatar's behavior and maintains synchronization between audio and video streams.

    The service supports:

    - Real-time avatar animation based on audio input
    - Voice activity detection for natural interactions
    - Interrupt handling for more natural conversations
    - Audio resampling for optimal quality
    - Automatic session management

    Args:
        api_key (str): AgentHuman API key for authentication
        session_request (NewSessionRequest, optional): Configuration for the AgentHuman session.
            Defaults to using the "avat_01KMZHXFPBVCXA5ATK85HCP8G1" avatar with "auto" aspect ratio.
    """

    def __init__(
        self,
        *,
        api_key: str,
        session_request: NewSessionRequest = NewSessionRequest(),
        transport: Optional[BaseTransport] = None,
        **kwargs,
    ) -> None:
        """Initialize the AgentHuman video service.

        Args:
            api_key: AgentHuman API key for authentication
            session_request: Configuration for the AgentHuman session. When ``aspect_ratio``
                is ``"auto"`` (the default), the nearest supported ratio is derived from the
                transport dimensions and used for the API request.
            **kwargs: Additional arguments passed to parent AIService
        """
        super().__init__(**kwargs)
        self._api_key = api_key
        self._client: Optional[AgentHumanClient] = None
        self._audio_buffer: Optional[bytearray] = None
        self._resampler = create_stream_resampler()
        self._other_participant_has_joined = False
        self._audio_chunk_size = 0
        self._transport = transport

        if not self._transport or not self._transport._params.video_out_enabled:
            raise ValueError("Transport must be provided and video output must be enabled")

        self._video_out_width = self._transport._params.video_out_width
        self._video_out_height = self._transport._params.video_out_height

        nearest_ratio = _nearest_aspect_ratio(self._video_out_width, self._video_out_height)

        if not _is_standard_ratio(self._video_out_width, self._video_out_height):
            logger.warning(
                f"Transport dimensions {self._video_out_width}x{self._video_out_height} do not closely "
                f"match any standard aspect ratio. Consider using one of: "
                f"{', '.join(_SUPPORTED_ASPECT_RATIOS)}. "
                f"Using nearest ratio '{nearest_ratio}'."
            )
        else:
            logger.debug(
                f"Transport dimensions {self._video_out_width}x{self._video_out_height} → "
                f"nearest aspect ratio: '{nearest_ratio}'"
            )

        if session_request.aspect_ratio == "auto":
            logger.info(
                f"aspect_ratio is 'auto'; resolved to '{nearest_ratio}' from transport dimensions "
                f"{self._video_out_width}x{self._video_out_height}"
            )
            self._session_request = session_request.model_copy(update={"aspect_ratio": nearest_ratio})
        else:
            if session_request.aspect_ratio != nearest_ratio:
                logger.warning(
                    f"Provided aspect_ratio '{session_request.aspect_ratio}' does not match the nearest "
                    f"ratio '{nearest_ratio}' for transport dimensions "
                    f"{self._video_out_width}x{self._video_out_height}. Proceeding with provided value."
                )
            self._session_request = session_request

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the AgentHuman video service with necessary configuration.

        Initializes the AgentHuman client, establishes connections, and prepares the service
        for audio/video processing. This includes setting up audio/video streams,
        configuring callbacks, and initializing the resampler.

        Args:
            setup: Configuration parameters for the frame processor.
        """
        await super().setup(setup)
        self._client = AgentHumanClient(
            api_key=self._api_key,
            params=TransportParams(
                audio_in_enabled=True,
                video_in_enabled=True,
                audio_out_enabled=True,
                audio_out_sample_rate=AGENTHUMAN_SAMPLE_RATE,
            ),
            session_request=self._session_request,
            callbacks=AgentHumanCallbacks(
                on_participant_connected=self._on_participant_connected,
                on_participant_disconnected=self._on_participant_disconnected,
            ),
        )
        await self._client.setup(setup)

    async def cleanup(self):
        """Clean up the service and release resources.

        Terminates the AgentHuman client session and cleans up associated resources.
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
        """Handle participant disconnected events.

        Does not call AgentHumanClient.stop() here: session teardown must go through
        ``stop`` / ``cancel`` → ``_end_conversation()`` so we do not end the API
        session twice or while the pipeline may still be active. Reset join state so
        a reconnect can attach capture again if the same flow emits another connect.
        """
        logger.info(f"Participant disconnected {participant_id}")
        self._other_participant_has_joined = False

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
        """Start the AgentHuman video service and initialize the avatar session.

        Creates necessary tasks for audio/video processing and establishes
        the connection with the AgentHuman service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._audio_chunk_size = int((AGENTHUMAN_SAMPLE_RATE * 0.04)) # 40 ms of audio
        await self._client.start(frame)

    async def stop(self, frame: EndFrame):
        """Stop the AgentHuman video service gracefully.

        Performs cleanup by ending the conversation and cancelling ongoing tasks
        in a controlled manner.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._end_conversation()

    async def cancel(self, frame: CancelFrame):
        """Cancel the AgentHuman video service.

        Performs an immediate termination of the service, cleaning up resources
        without waiting for ongoing operations to complete.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._end_conversation()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and coordinate avatar behavior.

        Handles different types of frames to manage avatar interactions:
        - UserStartedSpeakingFrame: Activates avatar's listening animation
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
            await self.push_frame(frame, direction)
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

        Manages the interruption flow
        """

        await self._client.interrupt()

    async def _end_conversation(self):
        """End the current conversation and reset state.

        Stops the AgentHuman client and cleans up conversation-specific resources.
        """
        self._other_participant_has_joined = False
        if self._client is not None:
            await self._client.stop()

    async def _handle_audio_frame(self, frame: OutputAudioRawFrame):
        """Queue an audio frame for processing.

        Places the audio frame in the processing queue for synchronized
        delivery to the AgentHuman service.

        Args:
            frame: The audio frame to process.
        """
        if self._audio_buffer is None:
            self._audio_buffer = bytearray()

        audio = await self._resampler.resample(
            frame.audio, frame.sample_rate, AGENTHUMAN_SAMPLE_RATE
        )

        self._audio_buffer.extend(audio)

        if len(self._audio_buffer) >= self._audio_chunk_size:
            await self._client.agent_speak(bytes(self._audio_buffer))
            self._audio_buffer = bytearray()