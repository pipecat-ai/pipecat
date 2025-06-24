import os
from functools import partial
from typing import Any, Awaitable, Callable, Mapping, Optional

import aiohttp
from daily.daily import AudioData
from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_default_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    StartInterruptionFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.services.daily import (
    DailyCallbacks,
    DailyParams,
    DailyTransportClient,
)


class TavusApi:
    """
    A helper class for interacting with the Tavus API (v2).
    """

    BASE_URL = "https://tavusapi.com/v2"
    MOCK_CONVERSATION_ID = "dev-conversation"
    MOCK_PERSONA_NAME = "TestTavusTransport"

    def __init__(self, api_key: str, session: aiohttp.ClientSession):
        """
        Initialize the TavusApi client.

        Args:
            api_key (str): Tavus API key.
            session (aiohttp.ClientSession): An aiohttp session for making HTTP requests.
        """
        self._api_key = api_key
        self._session = session
        self._headers = {"Content-Type": "application/json", "x-api-key": self._api_key}
        # Only for development
        self._dev_room_url = os.getenv("TAVUS_SAMPLE_ROOM_URL")

    async def create_conversation(self, replica_id: str, persona_id: str) -> dict:
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
        if conversation_id is None or conversation_id == self.MOCK_CONVERSATION_ID:
            return

        url = f"{self.BASE_URL}/conversations/{conversation_id}/end"
        async with self._session.post(url, headers=self._headers) as r:
            r.raise_for_status()
            logger.debug(f"Ended Tavus conversation {conversation_id}")

    async def get_persona_name(self, persona_id: str) -> str:
        if self._dev_room_url is not None:
            return self.MOCK_PERSONA_NAME

        url = f"{self.BASE_URL}/personas/{persona_id}"
        async with self._session.get(url, headers=self._headers) as r:
            r.raise_for_status()
            response = await r.json()
            logger.debug(f"Fetched Tavus persona: {response}")
            return response["persona_name"]


class TavusCallbacks(BaseModel):
    """Callback handlers for the Tavus events.

    Attributes:
        on_participant_joined: Called when a participant joins.
        on_participant_left: Called when a participant leaves.
    """

    on_participant_joined: Callable[[Mapping[str, Any]], Awaitable[None]]
    on_participant_left: Callable[[Mapping[str, Any], str], Awaitable[None]]


class TavusParams(DailyParams):
    """Configuration parameters for the Tavus transport."""

    audio_in_enabled: bool = True
    audio_out_enabled: bool = True
    microphone_out_enabled: bool = False


class TavusTransportClient:
    """
    A transport client that integrates a Pipecat Bot with the Tavus platform by managing
    conversation sessions using the Tavus API.

    This client uses `TavusApi` to interact with the Tavus backend services. When a conversation
    is started via `TavusApi`, Tavus provides a `roomURL` that can be used to connect the Pipecat Bot
    into the same virtual room where the TavusBot is operating.

    Args:
        bot_name (str): The name of the Pipecat bot instance.
        params (TavusParams): Optional parameters for Tavus operation. Defaults to `TavusParams()`.
        callbacks (TavusCallbacks): Callback handlers for Tavus-related events.
        api_key (str): API key for authenticating with Tavus API.
        replica_id (str): ID of the replica to use in the Tavus conversation.
        persona_id (str): ID of the Tavus persona. Defaults to "pipecat-stream", which signals Tavus to use
                          the TTS voice of the Pipecat bot instead of a Tavus persona voice.
        session (aiohttp.ClientSession): The aiohttp session for making async HTTP requests.
        sample_rate: Audio sample rate to be used by the client.
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
        self._bot_name = bot_name
        self._api = TavusApi(api_key, session)
        self._replica_id = replica_id
        self._persona_id = persona_id
        self._conversation_id: Optional[str] = None
        self._client: Optional[DailyTransportClient] = None
        self._callbacks = callbacks
        self._params = params

    async def _initialize(self) -> str:
        response = await self._api.create_conversation(self._replica_id, self._persona_id)
        self._conversation_id = response["conversation_id"]
        return response["conversation_url"]

    async def setup(self, setup: FrameProcessorSetup):
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
        try:
            await self._client.cleanup()
        except Exception as e:
            logger.exception(f"Exception during cleanup: {e}")

    async def _on_joined(self, data):
        logger.debug("TavusTransportClient joined!")

    async def _on_left(self):
        logger.debug("TavusTransportClient left!")

    async def _on_handle_callback(self, event_name, *args, **kwargs):
        logger.trace(f"[Callback] {event_name} called with args={args}, kwargs={kwargs}")

    async def get_persona_name(self) -> str:
        return await self._api.get_persona_name(self._persona_id)

    async def start(self, frame: StartFrame):
        logger.debug("TavusTransportClient start invoked!")
        await self._client.start(frame)
        await self._client.join()

    async def stop(self):
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
        await self._client.capture_participant_audio(
            participant_id, callback, audio_source, sample_rate, callback_interval_ms
        )

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        await self._client.send_message(frame)

    @property
    def out_sample_rate(self) -> int:
        return self._client.out_sample_rate

    @property
    def in_sample_rate(self) -> int:
        return self._client.in_sample_rate

    async def send_interrupt_message(self) -> None:
        transport_frame = TransportMessageUrgentFrame(
            message={
                "message_type": "conversation",
                "event_type": "conversation.interrupt",
                "conversation_id": self._conversation_id,
            }
        )
        await self.send_message(transport_frame)

    async def update_subscriptions(self, participant_settings=None, profile_settings=None):
        if not self._client:
            return

        await self._client.update_subscriptions(
            participant_settings=participant_settings, profile_settings=profile_settings
        )

    async def write_audio_frame(self, frame: OutputAudioRawFrame):
        if not self._client:
            return
        await self._client.write_audio_frame(frame)

    async def register_audio_destination(self, destination: str):
        if not self._client:
            return

        await self._client.register_audio_destination(destination)


class TavusInputTransport(BaseInputTransport):
    def __init__(
        self,
        client: TavusTransportClient,
        params: TransportParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params
        self._resampler = create_default_resampler()

        # Whether we have seen a StartFrame already.
        self._initialized = False

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        await self._client.setup(setup)

    async def cleanup(self):
        await super().cleanup()
        await self._client.cleanup()

    async def start(self, frame: StartFrame):
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        await self._client.start(frame)
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._client.stop()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._client.stop()

    async def start_capturing_audio(self, participant):
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
        frame = InputAudioRawFrame(
            audio=audio.audio_frames,
            sample_rate=audio.audio_frames,
            num_channels=audio.num_channels,
        )
        frame.transport_source = audio_source
        await self.push_audio_frame(frame)


class TavusOutputTransport(BaseOutputTransport):
    def __init__(
        self,
        client: TavusTransportClient,
        params: TransportParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params

        # Whether we have seen a StartFrame already.
        self._initialized = False
        # This is the custom track destination expected by Tavus
        self._transport_destination: Optional[str] = "stream"

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        await self._client.setup(setup)

    async def cleanup(self):
        await super().cleanup()
        await self._client.cleanup()

    async def start(self, frame: StartFrame):
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        await self._client.start(frame)

        if self._transport_destination:
            await self._client.register_audio_destination(self._transport_destination)

        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._client.stop()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._client.stop()

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        logger.info(f"TavusOutputTransport sending message {frame}")
        await self._client.send_message(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartInterruptionFrame):
            await self._handle_interruptions()

    async def _handle_interruptions(self):
        await self._client.send_interrupt_message()

    async def write_audio_frame(self, frame: OutputAudioRawFrame):
        # This is the custom track destination expected by Tavus
        frame.transport_destination = self._transport_destination
        await self._client.write_audio_frame(frame)

    async def register_audio_destination(self, destination: str):
        await self._client.register_audio_destination(destination)


class TavusTransport(BaseTransport):
    """
    Transport implementation for Tavus video calls.

    When used, the Pipecat bot joins the same virtual room as the Tavus Avatar and the user.
    This is achieved by using `TavusTransportClient`, which initiates the conversation via
    `TavusApi` and obtains a room URL that all participants connect to.

    Args:
        bot_name (str): The name of the Pipecat bot.
        session (aiohttp.ClientSession): aiohttp session used for async HTTP requests.
        api_key (str): Tavus API key for authentication.
        replica_id (str): ID of the replica model used for voice generation.
        persona_id (str): ID of the Tavus persona. Defaults to "pipecat-stream" to use the Pipecat TTS voice.
        params (TavusParams): Optional Tavus-specific configuration parameters.
        input_name (Optional[str]): Optional name for the input transport.
        output_name (Optional[str]): Optional name for the output transport.
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
        persona_name = await self._client.get_persona_name()
        if participant.get("info", {}).get("userName", "") != persona_name:
            await self._on_client_disconnected(participant)

    async def _on_participant_joined(self, participant):
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
        await self._client.update_subscriptions(
            participant_settings=participant_settings,
            profile_settings=profile_settings,
        )

    def input(self) -> FrameProcessor:
        if not self._input:
            self._input = TavusInputTransport(client=self._client, params=self._params)
        return self._input

    def output(self) -> FrameProcessor:
        if not self._output:
            self._output = TavusOutputTransport(client=self._client, params=self._params)
        return self._output

    async def _on_client_connected(self, participant: Any):
        await self._call_event_handler("on_client_connected", participant)

    async def _on_client_disconnected(self, participant: Any):
        await self._call_event_handler("on_client_disconnected", participant)
