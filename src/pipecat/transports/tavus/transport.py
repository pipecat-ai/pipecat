#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tavus transport implementation for Pipecat.

This module provides integration with the Tavus platform for creating conversational
AI applications with avatars. It manages conversation sessions and provides real-time
audio/video streaming capabilities through the Tavus API.
"""

import os
from functools import partial
from typing import Any, Awaitable, Callable, Mapping, Optional

import aiohttp
from daily.daily import AudioData
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
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


class TavusApi:
    """Helper class for interacting with the Tavus API (v2).

    Provides methods for creating and managing conversations with Tavus avatars,
    including conversation lifecycle management and persona information retrieval.
    """

    BASE_URL = "https://tavusapi.com/v2"
    MOCK_CONVERSATION_ID = "dev-conversation"
    MOCK_PERSONA_NAME = "TestTavusTransport"

    def __init__(self, api_key: str, session: aiohttp.ClientSession):
        """Initialize the TavusApi client.

        Args:
            api_key: Tavus API key for authentication.
            session: An aiohttp session for making HTTP requests.
        """
        self._api_key = api_key
        self._session = session
        self._headers = {"Content-Type": "application/json", "x-api-key": self._api_key}
        # Only for development
        self._dev_room_url = os.getenv("TAVUS_SAMPLE_ROOM_URL")

    async def create_conversation(self, replica_id: str, persona_id: str) -> dict:
        """Create a new conversation with the specified replica and persona.

        Args:
            replica_id: ID of the replica to use in the conversation.
            persona_id: ID of the persona to use in the conversation.

        Returns:
            Dictionary containing conversation_id and conversation_url.
        """
        if self._dev_room_url:
            return {
                "conversation_id": self.MOCK_CONVERSATION_ID,
                "conversation_url": self._dev_room_url,
            }

        logger.debug(f"Creating Tavus conversation: replica={replica_id}, persona={persona_id}")
        url = f"{self.BASE_URL}/conversations"
        payload = {
            "replica_id": replica_id,
            "persona_id": persona_id,
        }
        async with self._session.post(url, headers=self._headers, json=payload) as r:
            r.raise_for_status()
            response = await r.json()
            logger.debug(f"Created Tavus conversation: {response}")
            return response

    async def end_conversation(self, conversation_id: str):
        """End an existing conversation.

        Args:
            conversation_id: ID of the conversation to end.
        """
        if conversation_id is None or conversation_id == self.MOCK_CONVERSATION_ID:
            return

        url = f"{self.BASE_URL}/conversations/{conversation_id}/end"
        async with self._session.post(url, headers=self._headers) as r:
            r.raise_for_status()
            logger.debug(f"Ended Tavus conversation {conversation_id}")

    async def get_persona_name(self, persona_id: str) -> str:
        """Get the name of a persona by ID.

        Args:
            persona_id: ID of the persona to retrieve.

        Returns:
            The name of the persona.
        """
        if self._dev_room_url is not None:
            return self.MOCK_PERSONA_NAME

        url = f"{self.BASE_URL}/personas/{persona_id}"
        async with self._session.get(url, headers=self._headers) as r:
            r.raise_for_status()
            response = await r.json()
            logger.debug(f"Fetched Tavus persona: {response}")
            return response["persona_name"]


class TavusCallbacks(BaseModel):
    """Callback handlers for Tavus events.

    Parameters:
        on_participant_joined: Called when a participant joins the conversation.
        on_participant_left: Called when a participant leaves the conversation.
    """

    on_participant_joined: Callable[[Mapping[str, Any]], Awaitable[None]]
    on_participant_left: Callable[[Mapping[str, Any], str], Awaitable[None]]


class TavusParams(DailyParams):
    """Configuration parameters for the Tavus transport.

    Parameters:
        audio_in_enabled: Whether to enable audio input from participants.
        audio_out_enabled: Whether to enable audio output to participants.
        microphone_out_enabled: Whether to enable microphone output track.
    """

    audio_in_enabled: bool = True
    audio_out_enabled: bool = True
    microphone_out_enabled: bool = False


class TavusTransportClient:
    """Transport client that integrates Pipecat with the Tavus platform.

    A transport client that integrates a Pipecat Bot with the Tavus platform by managing
    conversation sessions using the Tavus API.

    This client uses `TavusApi` to interact with the Tavus backend services. When a conversation
    is started via `TavusApi`, Tavus provides a `roomURL` that can be used to connect the Pipecat Bot
    into the same virtual room where the TavusBot is operating.
    """

    def __init__(
        self,
        *,
        bot_name: str,
        params: TavusParams = TavusParams(),
        callbacks: TavusCallbacks,
        api_key: str,
        replica_id: str,
        persona_id: str = "pipecat-stream",
        session: aiohttp.ClientSession,
    ) -> None:
        """Initialize the Tavus transport client.

        Args:
            bot_name: The name of the Pipecat bot instance.
            params: Optional parameters for Tavus operation.
            callbacks: Callback handlers for Tavus-related events.
            api_key: API key for authenticating with Tavus API.
            replica_id: ID of the replica to use in the Tavus conversation.
            persona_id: ID of the Tavus persona. Defaults to "pipecat-stream",
                which signals Tavus to use the TTS voice of the Pipecat bot
                instead of a Tavus persona voice.
            session: The aiohttp session for making async HTTP requests.
        """
        self._bot_name = bot_name
        self._api = TavusApi(api_key, session)
        self._replica_id = replica_id
        self._persona_id = persona_id
        self._conversation_id: Optional[str] = None
        self._client: Optional[DailyTransportClient] = None
        self._callbacks = callbacks
        self._params = params

    async def _initialize(self) -> str:
        """Initialize the conversation and return the room URL."""
        response = await self._api.create_conversation(self._replica_id, self._persona_id)
        self._conversation_id = response["conversation_id"]
        return response["conversation_url"]

    async def setup(self, setup: FrameProcessorSetup):
        """Setup the client and initialize the conversation.

        Args:
            setup: The frame processor setup configuration.
        """
        if self._conversation_id is not None:
            logger.debug(f"Conversation ID already defined: {self._conversation_id}")
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
            self._client = DailyTransportClient(
                room_url, None, "Pipecat", self._params, daily_callbacks, self._bot_name
            )
            await self._client.setup(setup)
        except Exception as e:
            logger.error(f"Failed to setup TavusTransportClient: {e}")
            await self._api.end_conversation(self._conversation_id)
            self._conversation_id = None

    async def cleanup(self):
        """Cleanup client resources."""
        try:
            await self._client.cleanup()
        except Exception as e:
            logger.exception(f"Exception during cleanup: {e}")

    async def _on_joined(self, data):
        """Handle joined event."""
        logger.debug("TavusTransportClient joined!")

    async def _on_left(self):
        """Handle left event."""
        logger.debug("TavusTransportClient left!")

    async def _on_handle_callback(self, event_name, *args, **kwargs):
        """Handle generic callback events."""
        logger.trace(f"[Callback] {event_name} called with args={args}, kwargs={kwargs}")

    async def get_persona_name(self) -> str:
        """Get the persona name from the API.

        Returns:
            The name of the current persona.
        """
        return await self._api.get_persona_name(self._persona_id)

    async def start(self, frame: StartFrame):
        """Start the client and join the room.

        Args:
            frame: The start frame containing initialization parameters.
        """
        logger.debug("TavusTransportClient start invoked!")
        await self._client.start(frame)
        await self._client.join()

    async def stop(self):
        """Stop the client and end the conversation."""
        await self._client.leave()
        await self._api.end_conversation(self._conversation_id)
        self._conversation_id = None

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
        await self._client.capture_participant_video(
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
        await self._client.capture_participant_audio(
            participant_id, callback, audio_source, sample_rate, callback_interval_ms
        )

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send a message to participants.

        Args:
            frame: The message frame to send.
        """
        await self._client.send_message(frame)

    @property
    def out_sample_rate(self) -> int:
        """Get the output sample rate.

        Returns:
            The output sample rate in Hz.
        """
        return self._client.out_sample_rate

    @property
    def in_sample_rate(self) -> int:
        """Get the input sample rate.

        Returns:
            The input sample rate in Hz.
        """
        return self._client.in_sample_rate

    async def send_interrupt_message(self) -> None:
        """Send an interrupt message to the conversation."""
        transport_frame = OutputTransportMessageUrgentFrame(
            message={
                "message_type": "conversation",
                "event_type": "conversation.interrupt",
                "conversation_id": self._conversation_id,
            }
        )
        await self.send_message(transport_frame)

    async def update_subscriptions(self, participant_settings=None, profile_settings=None):
        """Update subscription settings for participants.

        Args:
            participant_settings: Per-participant subscription settings.
            profile_settings: Global subscription profile settings.
        """
        if not self._client:
            return

        await self._client.update_subscriptions(
            participant_settings=participant_settings, profile_settings=profile_settings
        )

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the transport.

        Args:
            frame: The audio frame to write.

        Returns:
            True if the audio frame was written successfully, False otherwise.
        """
        if not self._client:
            return False
        return await self._client.write_audio_frame(frame)

    async def register_audio_destination(self, destination: str):
        """Register an audio destination for output.

        Args:
            destination: The destination identifier to register.
        """
        if not self._client:
            return

        await self._client.register_audio_destination(destination)


class TavusInputTransport(BaseInputTransport):
    """Input transport for receiving audio and events from Tavus conversations.

    Handles incoming audio streams from participants and manages audio capture
    from the Daily room connected to the Tavus conversation.
    """

    def __init__(
        self,
        client: TavusTransportClient,
        params: TransportParams,
        **kwargs,
    ):
        """Initialize the Tavus input transport.

        Args:
            client: The Tavus transport client instance.
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
            logger.info(
                f"TavusTransportClient start capturing audio for participant {participant['id']}"
            )
            await self._client.capture_participant_audio(
                participant_id=participant["id"],
                callback=self._on_participant_audio_data,
                sample_rate=self._client.in_sample_rate,
            )

    async def _on_participant_audio_data(
        self, participant_id: str, audio: AudioData, audio_source: str
    ):
        """Handle received participant audio data."""
        frame = InputAudioRawFrame(
            audio=audio.audio_frames,
            sample_rate=audio.audio_frames,
            num_channels=audio.num_channels,
        )
        frame.transport_source = audio_source
        await self.push_audio_frame(frame)


class TavusOutputTransport(BaseOutputTransport):
    """Output transport for sending audio and events to Tavus conversations.

    Handles outgoing audio streams to participants and manages the custom
    audio track expected by the Tavus platform.
    """

    def __init__(
        self,
        client: TavusTransportClient,
        params: TransportParams,
        **kwargs,
    ):
        """Initialize the Tavus output transport.

        Args:
            client: The Tavus transport client instance.
            params: Transport configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params

        # Whether we have seen a StartFrame already.
        self._initialized = False
        # This is the custom track destination expected by Tavus
        self._transport_destination: Optional[str] = "stream"

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
        logger.info(f"TavusOutputTransport sending message {frame}")
        await self._client.send_message(frame)

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

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the Tavus transport.

        Args:
            frame: The audio frame to write.

        Returns:
            True if the audio frame was written successfully, False otherwise.
        """
        # This is the custom track destination expected by Tavus
        frame.transport_destination = self._transport_destination
        return await self._client.write_audio_frame(frame)

    async def register_audio_destination(self, destination: str):
        """Register an audio destination.

        Args:
            destination: The destination identifier to register.
        """
        await self._client.register_audio_destination(destination)


class TavusTransport(BaseTransport):
    """Transport implementation for Tavus video calls.

    When used, the Pipecat bot joins the same virtual room as the Tavus Avatar and the user.
    This is achieved by using `TavusTransportClient`, which initiates the conversation via
    `TavusApi` and obtains a room URL that all participants connect to.
    """

    def __init__(
        self,
        bot_name: str,
        session: aiohttp.ClientSession,
        api_key: str,
        replica_id: str,
        persona_id: str = "pipecat-stream",
        params: TavusParams = TavusParams(),
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        """Initialize the Tavus transport.

        Args:
            bot_name: The name of the Pipecat bot.
            session: aiohttp session used for async HTTP requests.
            api_key: Tavus API key for authentication.
            replica_id: ID of the replica model used for voice generation.
            persona_id: ID of the Tavus persona. Defaults to "pipecat-stream"
                to use the Pipecat TTS voice.
            params: Optional Tavus-specific configuration parameters.
            input_name: Optional name for the input transport.
            output_name: Optional name for the output transport.
        """
        super().__init__(input_name=input_name, output_name=output_name)
        self._params = params

        callbacks = TavusCallbacks(
            on_participant_joined=self._on_participant_joined,
            on_participant_left=self._on_participant_left,
        )
        self._client = TavusTransportClient(
            bot_name="Pipecat",
            callbacks=callbacks,
            api_key=api_key,
            replica_id=replica_id,
            persona_id=persona_id,
            session=session,
            params=params,
        )
        self._input: Optional[TavusInputTransport] = None
        self._output: Optional[TavusOutputTransport] = None
        self._tavus_participant_id = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")

    async def _on_participant_left(self, participant, reason):
        """Handle participant left events."""
        persona_name = await self._client.get_persona_name()
        if participant.get("info", {}).get("userName", "") != persona_name:
            await self._on_client_disconnected(participant)

    async def _on_participant_joined(self, participant):
        """Handle participant joined events."""
        # get persona, look up persona_name, set this as the bot name to ignore
        persona_name = await self._client.get_persona_name()

        # Ignore the Tavus replica's microphone
        if participant.get("info", {}).get("userName", "") == persona_name:
            self._tavus_participant_id = participant["id"]
        else:
            await self._on_client_connected(participant)
            if self._tavus_participant_id:
                logger.debug(f"Ignoring {self._tavus_participant_id}'s microphone")
                await self.update_subscriptions(
                    participant_settings={
                        self._tavus_participant_id: {
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
            The Tavus input transport instance.
        """
        if not self._input:
            self._input = TavusInputTransport(client=self._client, params=self._params)
        return self._input

    def output(self) -> FrameProcessor:
        """Get the output transport for sending media and events.

        Returns:
            The Tavus output transport instance.
        """
        if not self._output:
            self._output = TavusOutputTransport(client=self._client, params=self._params)
        return self._output

    async def _on_client_connected(self, participant: Any):
        """Handle client connected events."""
        await self._call_event_handler("on_client_connected", participant)

    async def _on_client_disconnected(self, participant: Any):
        """Handle client disconnected events."""
        await self._call_event_handler("on_client_disconnected", participant)
