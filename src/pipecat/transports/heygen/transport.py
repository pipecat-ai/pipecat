#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""HeyGen implementation for Pipecat.

This module provides integration with the HeyGen platform for creating conversational
AI applications with avatars. It manages conversation sessions and provides real-time
audio/video streaming capabilities through the HeyGen API.

The module consists of three main components:
- HeyGenInputTransport: Handles incoming audio and events from HeyGen conversations
- HeyGenOutputTransport: Manages outgoing audio and events to HeyGen conversations
- HeyGenTransport: Main transport implementation that coordinates input/output transports
"""

from typing import Any, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.services.heygen.api import NewSessionRequest
from pipecat.services.heygen.client import HeyGenCallbacks, HeyGenClient
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams


class HeyGenInputTransport(BaseInputTransport):
    """Input transport for receiving audio and events from HeyGen conversations.

    Handles incoming audio streams from participants and manages audio capture
    from the Daily room connected to the HeyGen conversation.
    """

    def __init__(
        self,
        client: HeyGenClient,
        params: TransportParams,
        **kwargs,
    ):
        """Initialize the HeyGen input transport.

        Args:
            client: The HeyGen transport client instance.
            params: Transport configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params
        # Whether we have seen a StartFrame already.
        self._initialized = False

    async def setup(self, setup: FrameProcessorSetup):
        """Setup the input transport.

        Args:
            setup: The frame processor setup configuration.
        """
        await super().setup(setup)
        await self._client.setup(setup)

    async def cleanup(self):
        """Cleanup input transport resources."""
        await super().cleanup()
        await self._client.cleanup()

    async def start(self, frame: StartFrame):
        """Start the input transport.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the input transport.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._client.stop()

    async def cancel(self, frame: CancelFrame):
        """Cancel the input transport.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._client.stop()

    async def start_capturing_audio(self, participant_id: str):
        """Start capturing audio from a participant.

        Args:
            participant_id: The participant to capture audio from.
        """
        if self._params.audio_in_enabled:
            logger.info(f"HeyGenTransport start capturing audio for participant {participant_id}")
            await self._client.capture_participant_audio(
                participant_id, self._on_participant_audio_data
            )

    async def _on_participant_audio_data(self, audio_frame: AudioRawFrame):
        """Handle received participant audio data."""
        frame = InputAudioRawFrame(
            audio=audio_frame.audio,
            sample_rate=audio_frame.sample_rate,
            num_channels=audio_frame.num_channels,
        )
        await self.push_audio_frame(frame)


class HeyGenOutputTransport(BaseOutputTransport):
    """Output transport for sending audio and events to HeyGen conversations.

    Handles outgoing audio streams to participants and manages the custom
    audio track expected by the HeyGen platform.
    """

    def __init__(
        self,
        client: HeyGenClient,
        params: TransportParams,
        **kwargs,
    ):
        """Initialize the HeyGen output transport.

        Args:
            client: The HeyGen transport client instance.
            params: Transport configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params

        # Whether we have seen a StartFrame already.
        self._initialized = False
        self._event_id = None

    async def setup(self, setup: FrameProcessorSetup):
        """Setup the output transport.

        Args:
            setup: The frame processor setup configuration.
        """
        await super().setup(setup)
        await self._client.setup(setup)

    async def cleanup(self):
        """Cleanup output transport resources."""
        await super().cleanup()
        await self._client.cleanup()

    async def start(self, frame: StartFrame):
        """Start the output transport.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True
        await self._client.start(frame, self.audio_chunk_size)
        await self.set_transport_ready(frame)
        self._client.transport_ready()

    async def stop(self, frame: EndFrame):
        """Stop the output transport.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._client.stop()

    async def cancel(self, frame: CancelFrame):
        """Cancel the output transport.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._client.stop()

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame to the next processor in the pipeline.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        # The BotStartedSpeakingFrame and BotStoppedSpeakingFrame are created inside BaseOutputTransport
        # This is a workaround, so we can more reliably be aware when the bot has started or stopped speaking
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, BotStartedSpeakingFrame):
                if self._event_id is not None:
                    logger.warning("self._event_id is already defined!")
                self._event_id = str(frame.id)
            elif isinstance(frame, BotStoppedSpeakingFrame):
                await self._client.agent_speak_end(self._event_id)
                self._event_id = None
        await super().push_frame(frame, direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and handle interruptions.

        Handles various types of frames including interruption events and user speaking states.
        Updates the HeyGen client state based on the received frames.

        Args:
            frame: The frame to process
            direction: The direction of frame flow in the pipeline

        Note:
            Special handling is implemented for:
            - InterruptionFrame: Triggers interruption of current speech
            - UserStartedSpeakingFrame: Initiates agent listening mode
            - UserStoppedSpeakingFrame: Stops agent listening mode
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            await self._client.interrupt(self._event_id)
            await self.push_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            await self._client.start_agent_listening()
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._client.stop_agent_listening()
            await self.push_frame(frame, direction)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the HeyGen transport.

        Args:
            frame: The audio frame to write.
        """
        await self._client.agent_speak(bytes(frame.audio), self._event_id)
        return True


class HeyGenParams(TransportParams):
    """Configuration parameters for the HeyGen transport.

    Parameters:
        audio_in_enabled: Whether to enable audio input from participants.
        audio_out_enabled: Whether to enable audio output to participants.
    """

    audio_in_enabled: bool = True
    audio_out_enabled: bool = True


class HeyGenTransport(BaseTransport):
    """Transport implementation for HeyGen video calls.

    When used, the Pipecat bot joins the same virtual room as the HeyGen Avatar and the user.
    This is achieved by using `HeyGenTransport`, which initiates the conversation via
    `HeyGenApi` and obtains a room URL that all participants connect to.
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        params: HeyGenParams = HeyGenParams(),
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        session_request: NewSessionRequest = NewSessionRequest(
            avatar_id="Shawn_Therapist_public",
            version="v2",
        ),
    ):
        """Initialize the HeyGen transport.

        Sets up a new HeyGen transport instance with the specified configuration for
        handling video calls between the Pipecat bot and HeyGen Avatar.

        Args:
            session: aiohttp session for making async HTTP requests
            api_key: HeyGen API key for authentication
            params: HeyGen-specific configuration parameters (default: HeyGenParams())
            input_name: Optional custom name for the input transport
            output_name: Optional custom name for the output transport
            session_request: Configuration for the HeyGen session (default: uses Shawn_Therapist_public avatar)

        Note:
            The transport will automatically join the same virtual room as the HeyGen Avatar
            and user through the HeyGenClient, which handles session initialization via HeyGenApi.
        """
        super().__init__(input_name=input_name, output_name=output_name)
        self._params = params
        self._client = HeyGenClient(
            api_key=api_key,
            session=session,
            params=params,
            session_request=session_request,
            callbacks=HeyGenCallbacks(
                on_participant_connected=self._on_participant_connected,
                on_participant_disconnected=self._on_participant_disconnected,
            ),
        )
        self._input: Optional[HeyGenInputTransport] = None
        self._output: Optional[HeyGenOutputTransport] = None
        self._HeyGen_participant_id = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")

    async def _on_participant_disconnected(self, participant_id: str):
        logger.debug(f"HeyGen participant {participant_id} disconnected")
        if participant_id != "heygen":
            await self._on_client_disconnected(participant_id)

    async def _on_participant_connected(self, participant_id: str):
        logger.debug(f"HeyGen participant {participant_id} connected")
        if participant_id != "heygen":
            await self._on_client_connected(participant_id)
            if self._input:
                await self._input.start_capturing_audio(participant_id)

    def input(self) -> FrameProcessor:
        """Get the input transport for receiving media and events.

        Returns:
            The HeyGen input transport instance.
        """
        if not self._input:
            self._input = HeyGenInputTransport(client=self._client, params=self._params)
        return self._input

    def output(self) -> FrameProcessor:
        """Get the output transport for sending media and events.

        Returns:
            The HeyGen output transport instance.
        """
        if not self._output:
            self._output = HeyGenOutputTransport(client=self._client, params=self._params)
        return self._output

    async def _on_client_connected(self, participant: Any):
        """Handle client connected events."""
        await self._call_event_handler("on_client_connected", participant)

    async def _on_client_disconnected(self, participant: Any):
        """Handle client disconnected events."""
        await self._call_event_handler("on_client_disconnected", participant)
