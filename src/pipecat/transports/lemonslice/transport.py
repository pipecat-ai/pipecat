#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LemonSlice transport for Pipecat.

This module adds LemonSlice avatars to Daily rooms, enabling
real-time voice conversations with synchronized avatars.
"""

from collections.abc import Awaitable, Callable, Mapping
from functools import partial
from typing import Any

import aiohttp
from daily.daily import AudioData
from loguru import logger
from pydantic import BaseModel, ConfigDict

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import (
    DailyCallbacks,
    DailyParams,
    DailyTransportClient,
)
from pipecat.transports.lemonslice.api import LemonSliceApi


class LemonSliceNewSessionRequest(BaseModel):
    """Request model for creating a new LemonSlice session.

    Parameters:
        agent_image_url: URL to an agent image. Provide either agent_id or agent_image_url.
        agent_id: ID of a LemonSlice agent. Provide either agent_id or agent_image_url.
        agent_prompt: A high-level system prompt that subtly influences the avatar's movements,
            expressions, and emotional demeanor.
        idle_timeout: Idle timeout in seconds.
        daily_room_url: Daily room URL to use for the session.
        daily_token: Daily token for authenticating with the room.
        lemonslice_properties: Additional connection properties to pass to the session.
        api_url: Override the LemonSlice API URL.
    """

    model_config = ConfigDict(extra="allow")

    agent_image_url: str | None = None
    agent_id: str | None = None
    agent_prompt: str | None = None
    idle_timeout: int | None = None
    daily_room_url: str | None = None
    daily_token: str | None = None
    lemonslice_properties: dict | None = None
    api_url: str | None = None


class LemonSliceCallbacks(BaseModel):
    """Callback handlers for LemonSlice events.

    Parameters:
        on_participant_joined: Called when a participant joins the conversation.
        on_participant_left: Called when a participant leaves the conversation.
    """

    on_participant_joined: Callable[[Mapping[str, Any]], Awaitable[None]]
    on_participant_left: Callable[[Mapping[str, Any], str], Awaitable[None]]


class LemonSliceParams(DailyParams):
    """Configuration parameters for the LemonSlice transport.

    Parameters:
        audio_in_enabled: Whether to enable audio input from participants.
        audio_out_enabled: Whether to enable audio output to participants.
        microphone_out_enabled: Whether to enable microphone output track.
    """

    audio_in_enabled: bool = True
    audio_out_enabled: bool = True
    microphone_out_enabled: bool = False


class LemonSliceTransportClient:
    """Transport client that integrates Pipecat with the LemonSlice platform.

    A transport client that integrates a Pipecat Bot with the LemonSlice platform by managing
    conversation sessions using the LemonSlice API.

    This client uses `LemonSliceApi` to interact with the LemonSlice backend. LemonSlice either provides
    a room URL where the avatar is already present, or adds the LemonSlice avatar to a Daily room
    the user supplies.
    """

    def __init__(
        self,
        *,
        bot_name: str,
        params: LemonSliceParams = LemonSliceParams(),
        callbacks: LemonSliceCallbacks,
        api_key: str,
        session_request: LemonSliceNewSessionRequest | None = None,
        session: aiohttp.ClientSession,
    ) -> None:
        """Initialize the LemonSlice transport client.

        Args:
            bot_name: The name of the Pipecat bot instance.
            params: Optional parameters for LemonSlice operation.
            callbacks: Callback handlers for LemonSlice-related events.
            api_key: API key for authenticating with LemonSlice API.
            session_request: Optional session creation parameters. If not provided, a default
                agent will be used.
            session: The aiohttp session for making async HTTP requests.
        """
        self._bot_name = bot_name
        self._api = LemonSliceApi(api_key, session)
        self._session_request = session_request or LemonSliceNewSessionRequest()
        self._session_id: str | None = None
        self._control_url: str | None = None
        self._daily_transport_client: DailyTransportClient | None = None
        self._callbacks = callbacks
        self._params = params

    async def _initialize(self) -> str:
        """Initialize the conversation and return the room URL."""
        connection_properties = dict(self._session_request.lemonslice_properties or {})
        extra_properties = self._session_request.model_extra
        response = await self._api.create_session(
            agent_image_url=self._session_request.agent_image_url,
            agent_id=self._session_request.agent_id,
            agent_prompt=self._session_request.agent_prompt,
            idle_timeout=self._session_request.idle_timeout,
            daily_room_url=self._session_request.daily_room_url,
            daily_token=self._session_request.daily_token,
            connection_properties=connection_properties if connection_properties else None,
            extra_properties=extra_properties if extra_properties else None,
            api_url=self._session_request.api_url,
        )
        self._session_id = response["session_id"]
        self._control_url = response["control_url"]
        return response["room_url"]

    async def setup(self, setup: FrameProcessorSetup):
        """Setup the client and initialize the conversation.

        Args:
            setup: The frame processor setup configuration.
        """
        if self._session_id is not None:
            logger.debug(f"Session ID already defined: {self._session_id}")
            return
        try:
            room_url = await self._initialize()
            daily_callbacks = DailyCallbacks(
                on_active_speaker_changed=partial(
                    self._on_handle_callback, "on_active_speaker_changed"
                ),
                on_joined=self._on_joined,
                on_left=self._on_left,
                on_before_leave=partial(self._on_handle_callback, "on_before_leave"),
                on_error=partial(self._on_handle_callback, "on_error"),
                on_app_message=partial(self._on_handle_callback, "on_app_message"),
                on_call_state_updated=partial(self._on_handle_callback, "on_call_state_updated"),
                on_client_connected=partial(self._on_handle_callback, "on_client_connected"),
                on_client_disconnected=partial(self._on_handle_callback, "on_client_disconnected"),
                on_dialin_connected=partial(self._on_handle_callback, "on_dialin_connected"),
                on_dialin_ready=partial(self._on_handle_callback, "on_dialin_ready"),
                on_dialin_stopped=partial(self._on_handle_callback, "on_dialin_stopped"),
                on_dialin_error=partial(self._on_handle_callback, "on_dialin_error"),
                on_dialin_warning=partial(self._on_handle_callback, "on_dialin_warning"),
                on_dialout_answered=partial(self._on_handle_callback, "on_dialout_answered"),
                on_dialout_connected=partial(self._on_handle_callback, "on_dialout_connected"),
                on_dialout_stopped=partial(self._on_handle_callback, "on_dialout_stopped"),
                on_dialout_error=partial(self._on_handle_callback, "on_dialout_error"),
                on_dialout_warning=partial(self._on_handle_callback, "on_dialout_warning"),
                on_dtmf_event=partial(self._on_handle_callback, "on_dtmf_event"),
                on_participant_joined=self._callbacks.on_participant_joined,
                on_participant_left=self._callbacks.on_participant_left,
                on_participant_updated=partial(self._on_handle_callback, "on_participant_updated"),
                on_transcription_message=partial(
                    self._on_handle_callback, "on_transcription_message"
                ),
                on_recording_started=partial(self._on_handle_callback, "on_recording_started"),
                on_recording_stopped=partial(self._on_handle_callback, "on_recording_stopped"),
                on_recording_error=partial(self._on_handle_callback, "on_recording_error"),
                on_transcription_stopped=partial(
                    self._on_handle_callback, "on_transcription_stopped"
                ),
                on_transcription_error=partial(self._on_handle_callback, "on_transcription_error"),
            )
            self._daily_transport_client = DailyTransportClient(
                room_url, None, self._bot_name, self._params, daily_callbacks, "LemonSlicePipecat"
            )
            await self._daily_transport_client.setup(setup)
        except Exception as e:
            logger.error(f"Failed to setup LemonSliceTransportClient: {e}")
            if self._session_id and self._control_url:
                await self._api.end_session(self._session_id, self._control_url)
            self._session_id = None
            self._control_url = None
            raise

    async def cleanup(self):
        """Cleanup client resources."""
        try:
            if self._daily_transport_client:
                await self._daily_transport_client.cleanup()
        except Exception as e:
            logger.error(f"Exception during cleanup: {e}")

    async def _on_joined(self, data):
        """Handle joined event."""
        logger.debug("LemonSliceTransportClient joined!")

    async def _on_left(self):
        """Handle left event."""
        logger.debug("LemonSliceTransportClient left!")

    async def _on_handle_callback(self, event_name, *args, **kwargs):
        """Handle generic callback events."""
        logger.trace(f"[Callback] {event_name} called with args={args}, kwargs={kwargs}")

    async def get_bot_name(self) -> str:
        """Get the name of the LemonSlice participant.

        Returns:
            The name of the LemonSlice participant.
        """
        return "LemonSlice"

    async def start(self, frame: StartFrame):
        """Start the client and join the room.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await self._daily_transport_client.start(frame)
        await self._daily_transport_client.join()

    async def stop(self):
        """Stop the client and end the conversation."""
        await self._daily_transport_client.leave()
        if self._session_id and self._control_url:
            await self._api.end_session(self._session_id, self._control_url)
        self._session_id = None
        self._control_url = None

    async def capture_participant_video(
        self,
        participant_id: str,
        callback: Callable,
        framerate: int = 30,
        video_source: str = "camera",
        color_format: str = "RGB",
    ):
        """Capture video from a participant.

        Args:
            participant_id: ID of the participant to capture video from.
            callback: Callback function to handle video frames.
            framerate: Desired framerate for video capture.
            video_source: Video source to capture from.
            color_format: Color format for video frames.
        """
        await self._daily_transport_client.capture_participant_video(
            participant_id, callback, framerate, video_source, color_format
        )

    async def capture_participant_audio(
        self,
        participant_id: str,
        callback: Callable,
        audio_source: str = "microphone",
        sample_rate: int = 16000,
        callback_interval_ms: int = 20,
    ):
        """Capture audio from a participant.

        Args:
            participant_id: ID of the participant to capture audio from.
            callback: Callback function to handle audio data.
            audio_source: Audio source to capture from.
            sample_rate: Desired sample rate for audio capture.
            callback_interval_ms: Interval between audio callbacks in milliseconds.
        """
        await self._daily_transport_client.capture_participant_audio(
            participant_id, callback, audio_source, sample_rate, callback_interval_ms
        )

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send a message to participants.

        Args:
            frame: The message frame to send.
        """
        await self._daily_transport_client.send_message(frame)

    @property
    def out_sample_rate(self) -> int:
        """Get the output sample rate.

        Returns:
            The output sample rate in Hz.
        """
        return self._daily_transport_client.out_sample_rate

    @property
    def in_sample_rate(self) -> int:
        """Get the input sample rate.

        Returns:
            The input sample rate in Hz.
        """
        return self._daily_transport_client.in_sample_rate

    async def send_interrupt_message(self) -> None:
        """Send an interrupt message to the LemonSlice session."""
        logger.debug("Sending interrupt message")
        transport_frame = OutputTransportMessageUrgentFrame(
            message={
                "event": "interrupt",
                "session_id": self._session_id,
            }
        )
        await self.send_message(transport_frame)

    async def send_response_started_message(self) -> None:
        """Send a response_started message to the LemonSlice session."""
        logger.trace("Sending response_started message")
        transport_frame = OutputTransportMessageUrgentFrame(
            message={
                "event": "response_started",
                "session_id": self._session_id,
            }
        )
        await self.send_message(transport_frame)

    async def send_response_finished_message(self) -> None:
        """Send a response_finished message to the LemonSlice session."""
        logger.trace("Sending response_finished message")
        transport_frame = OutputTransportMessageUrgentFrame(
            message={
                "event": "response_finished",
                "session_id": self._session_id,
            }
        )
        await self.send_message(transport_frame)

    async def update_subscriptions(self, participant_settings=None, profile_settings=None):
        """Update subscription settings for participants.

        Args:
            participant_settings: Per-participant subscription settings.
            profile_settings: Global subscription profile settings.
        """
        if not self._daily_transport_client:
            return

        await self._daily_transport_client.update_subscriptions(
            participant_settings=participant_settings, profile_settings=profile_settings
        )

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the transport.

        Args:
            frame: The audio frame to write.

        Returns:
            True if the audio frame was written successfully, False otherwise.
        """
        if not self._daily_transport_client:
            return False

        return await self._daily_transport_client.write_audio_frame(frame)

    async def register_audio_destination(self, destination: str):
        """Register an audio destination for output.

        Args:
            destination: The destination identifier to register.
        """
        if not self._daily_transport_client:
            return

        await self._daily_transport_client.register_audio_destination(destination)


class LemonSliceInputTransport(BaseInputTransport):
    """Input transport for receiving audio and events from LemonSlice.

    Handles incoming audio streams from participants and manages audio capture
    from the Daily room connected to LemonSlice.
    """

    def __init__(
        self,
        client: LemonSliceTransportClient,
        params: TransportParams,
        **kwargs,
    ):
        """Initialize the LemonSlice input transport.

        Args:
            client: The LemonSlice transport client instance.
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

        await self._client.start(frame)
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

    async def start_capturing_audio(self, participant):
        """Start capturing audio from a participant.

        Args:
            participant: The participant to capture audio from.
        """
        if self._params.audio_in_enabled:
            logger.debug(
                f"LemonSliceTransportClient start capturing audio for participant {participant['id']}"
            )
            await self._client.capture_participant_audio(
                participant_id=participant["id"],
                callback=self._on_participant_audio_data,
                sample_rate=self._client.in_sample_rate,
            )

    async def _on_participant_audio_data(
        self, participant_id: str, audio: AudioData, audio_source: str
    ):
        """Handle received participant audio data.

        Args:
            participant_id: ID of the participant who sent the audio.
            audio: The audio data from the participant.
            audio_source: The source of the audio (e.g., microphone).
        """
        frame = InputAudioRawFrame(
            audio=audio.audio_frames,
            sample_rate=audio.sample_rate,
            num_channels=audio.num_channels,
        )
        frame.transport_source = audio_source
        await self.push_audio_frame(frame)


class LemonSliceOutputTransport(BaseOutputTransport):
    """Output transport for sending audio and events to LemonSlice.

    Handles outgoing audio streams to participants and manages the custom
    audio track expected by the LemonSlice platform.
    """

    def __init__(
        self,
        client: LemonSliceTransportClient,
        params: TransportParams,
        **kwargs,
    ):
        """Initialize the LemonSlice output transport.

        Args:
            client: The LemonSlice transport client instance.
            params: Transport configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params

        # Whether we have seen a StartFrame already.
        self._initialized = False
        # This is the custom track destination expected by LemonSlice
        self._transport_destination: str | None = "stream"

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

        await self._client.start(frame)

        if self._transport_destination:
            await self._client.register_audio_destination(self._transport_destination)

        await self.set_transport_ready(frame)

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

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send a message to participants.

        Args:
            frame: The message frame to send.
        """
        logger.trace(f"LemonSliceTransport sending message {frame}")
        await self._client.send_message(frame)

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
                await self._handle_response_started()
            if isinstance(frame, BotStoppedSpeakingFrame):
                await self._handle_response_finished()
        await super().push_frame(frame, direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and handle interruptions.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            await self._handle_interruptions()

    async def _handle_interruptions(self):
        """Handle interruption events by sending interrupt message."""
        await self._client.send_interrupt_message()

    async def _handle_response_started(self):
        """Handle bot started speaking events by sending response_started message."""
        await self._client.send_response_started_message()

    async def _handle_response_finished(self):
        """Handle tts response stopped events by sending response_finished message."""
        await self._client.send_response_finished_message()

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the LemonSlice transport.

        Args:
            frame: The audio frame to write.

        Returns:
            True if the audio frame was written successfully, False otherwise.
        """
        # This is the custom track destination expected by LemonSlice
        frame.transport_destination = self._transport_destination
        return await self._client.write_audio_frame(frame)

    async def register_audio_destination(self, destination: str):
        """Register an audio destination.

        Args:
            destination: The destination identifier to register.
        """
        await self._client.register_audio_destination(destination)


class LemonSliceTransport(BaseTransport):
    """Transport implementation to add a LemonSlice avatar to Daily calls.

    When used, the Pipecat bot joins the same virtual room as the LemonSlice Avatar and the user.
    This is achieved by using `LemonSliceTransportClient`, which initiates the conversation via
    `LemonSliceApi` and obtains a room URL that all participants connect to.

    Event handlers available:

    - on_client_connected(transport, participant): Participant connected to the session
    - on_client_disconnected(transport, participant): Participant disconnected from the session
    - on_avatar_connected(transport, participant): LemonSlice avatar connected to the session
    - on_avatar_disconnected(transport, participant, reason): LemonSlice avatar disconnected from the session

    Example::

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, participant):
            ...
    """

    def __init__(
        self,
        bot_name: str,
        session: aiohttp.ClientSession,
        api_key: str,
        session_request: LemonSliceNewSessionRequest | None = None,
        params: LemonSliceParams = LemonSliceParams(),
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        """Initialize the LemonSlice transport.

        Args:
            bot_name: The name of the Pipecat bot.
            session: aiohttp session used for async HTTP requests.
            api_key: LemonSlice API key for authentication.
            session_request: Optional session creation parameters. If not provided, a default
                agent will be used.
            params: Optional LemonSlice-specific configuration parameters.
            input_name: Optional name for the input transport.
            output_name: Optional name for the output transport.
        """
        super().__init__(input_name=input_name, output_name=output_name)
        self._params = params

        callbacks = LemonSliceCallbacks(
            on_participant_joined=self._on_participant_joined,
            on_participant_left=self._on_participant_left,
        )
        self._client = LemonSliceTransportClient(
            bot_name=bot_name,
            callbacks=callbacks,
            api_key=api_key,
            session_request=session_request,
            session=session,
            params=params,
        )
        self._input: LemonSliceInputTransport | None = None
        self._output: LemonSliceOutputTransport | None = None
        self._lemonslice_participant_id = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_avatar_connected")
        self._register_event_handler("on_avatar_disconnected")

    async def _on_participant_left(self, participant, reason):
        """Handle participant left events."""
        ls_bot_name = await self._client.get_bot_name()
        if participant.get("info", {}).get("userName", "") == ls_bot_name:
            await self._on_avatar_disconnected(participant, reason)
        else:
            await self._on_client_disconnected(participant)

    async def _on_participant_joined(self, participant):
        """Handle participant joined events."""
        ls_bot_name = await self._client.get_bot_name()

        # Ignore the LemonSlice bot's microphone
        if participant.get("info", {}).get("userName", "") == ls_bot_name:
            self._lemonslice_participant_id = participant["id"]
            await self._on_avatar_connected(participant)
        else:
            await self._on_client_connected(participant)
            if self._lemonslice_participant_id:
                logger.debug(f"Ignoring {self._lemonslice_participant_id}'s microphone")
                await self.update_subscriptions(
                    participant_settings={
                        self._lemonslice_participant_id: {
                            "media": {"microphone": "unsubscribed"},
                        }
                    }
                )
            if self._input:
                await self._input.start_capturing_audio(participant)

    async def update_subscriptions(self, participant_settings=None, profile_settings=None):
        """Update subscription settings for participants.

        Args:
            participant_settings: Per-participant subscription settings.
            profile_settings: Global subscription profile settings.
        """
        await self._client.update_subscriptions(
            participant_settings=participant_settings,
            profile_settings=profile_settings,
        )

    def input(self) -> FrameProcessor:
        """Get the input transport for receiving media and events.

        Returns:
            The LemonSlice input transport instance.
        """
        if not self._input:
            self._input = LemonSliceInputTransport(client=self._client, params=self._params)
        return self._input

    def output(self) -> FrameProcessor:
        """Get the output transport for sending media and events.

        Returns:
            The LemonSlice output transport instance.
        """
        if not self._output:
            self._output = LemonSliceOutputTransport(client=self._client, params=self._params)
        return self._output

    async def _on_avatar_connected(self, participant: Any):
        """Handle avatar connected events."""
        await self._call_event_handler("on_avatar_connected", participant)

    async def _on_avatar_disconnected(self, participant: Any, reason: str):
        """Handle avatar disconnected events."""
        await self._call_event_handler("on_avatar_disconnected", participant, reason)

    async def _on_client_connected(self, participant: Any):
        """Handle client connected events."""
        await self._call_event_handler("on_client_connected", participant)

    async def _on_client_disconnected(self, participant: Any):
        """Handle client disconnected events."""
        await self._call_event_handler("on_client_disconnected", participant)
