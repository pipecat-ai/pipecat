import asyncio
import base64
import time
from functools import partial
from typing import Any, Awaitable, Callable, Mapping, Optional

import aiohttp
from daily.daily import AudioData
from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_default_resampler
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
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

    async def create_conversation(self, replica_id: str, persona_id: str) -> dict:
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
        if conversation_id is None:
            return

        url = f"{self.BASE_URL}/conversations/{conversation_id}/end"
        async with self._session.post(url, headers=self._headers) as r:
            r.raise_for_status()
            logger.debug(f"Ended Tavus conversation {conversation_id}")

    async def get_persona_name(self, persona_id: str) -> str:
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
        persona_id (str): ID of the Tavus persona. Defaults to "pipecat0", which signals Tavus to use
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
        persona_id: str = "pipecat0",  # Use `pipecat0` so that your TTS voice is used in place of the Tavus persona
        session: aiohttp.ClientSession,
    ) -> None:
        self._bot_name = bot_name
        self._api = TavusApi(api_key, session)
        self._replica_id = replica_id
        self._persona_id = persona_id
        self._conversation_id: Optional[str] = None
        self._other_participant_has_joined = False
        self._client: Optional[DailyTransportClient] = None
        self._callbacks = callbacks
        self._params = params

    async def _initialize(self) -> str:
        response = await self._api.create_conversation(self._replica_id, self._persona_id)
        self._conversation_id = response["conversation_id"]
        return response["conversation_url"]

    async def setup(self, setup: FrameProcessorSetup):
        if self._conversation_id is not None:
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

    async def cleanup(self):
        if self._client is None:
            return
        await self._client.cleanup()
        self._client = None

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

    async def encode_audio_and_send(self, audio: bytes, done: bool, inference_id: str):
        """Encodes audio to base64 and sends it to Tavus"""
        audio_base64 = base64.b64encode(audio).decode("utf-8")
        await self._send_audio_message(audio_base64, done=done, inference_id=inference_id)

    async def send_interrupt_message(self) -> None:
        transport_frame = TransportMessageUrgentFrame(
            message={
                "message_type": "conversation",
                "event_type": "conversation.interrupt",
                "conversation_id": self._conversation_id,
            }
        )
        await self.send_message(transport_frame)

    async def _send_audio_message(self, audio_base64: str, done: bool, inference_id: str):
        transport_frame = TransportMessageUrgentFrame(
            message={
                "message_type": "conversation",
                "event_type": "conversation.echo",
                "conversation_id": self._conversation_id,
                "properties": {
                    "modality": "audio",
                    "inference_id": inference_id,
                    "audio": audio_base64,
                    "done": done,
                    "sample_rate": self.out_sample_rate,
                },
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
        self._samples_sent = 0
        self._start_time = None
        self._current_idx_str: Optional[str] = None

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

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        logger.info(f"TavusOutputTransport sending message {frame}")
        await self._client.send_message(frame)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        # The BotStartedSpeakingFrame and BotStoppedSpeakingFrame are created inside BaseOutputTransport
        # so TavusOutputTransport never receives these frames.
        # This is a workaround, so we can more reliably be aware when the bot has started or stopped speaking
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, BotStartedSpeakingFrame):
                if self._current_idx_str is not None:
                    logger.warning("TavusOutputTransport self._current_idx_str is already defined!")
                self._current_idx_str = str(frame.id)
                self._start_time = time.time()
                self._samples_sent = 0
            elif isinstance(frame, BotStoppedSpeakingFrame):
                silence = b"\x00" * self.audio_chunk_size
                await self._client.encode_audio_and_send(silence, True, self._current_idx_str)
                self._current_idx_str = None
        await super().push_frame(frame, direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartInterruptionFrame):
            await self._handle_interruptions()

    async def _handle_interruptions(self):
        await self._client.send_interrupt_message()

    async def write_audio_frame(self, frame: OutputAudioRawFrame):
        # Compute wait time for synchronization
        wait = self._start_time + (self._samples_sent / self.sample_rate) - time.time()
        if wait > 0:
            logger.trace(f"TavusOutputTransport write_audio_frame wait: {wait}")
            await asyncio.sleep(wait)

        if self._current_idx_str is None:
            logger.warning("TavusOutputTransport self._current_idx_str not defined yet!")
            return

        await self._client.encode_audio_and_send(frame.audio, False, self._current_idx_str)

        # Update timestamp based on number of samples sent
        self._samples_sent += len(frame.audio) // 2  # 2 bytes per sample (16-bit)


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
        persona_id (str): ID of the Tavus persona. Defaults to "pipecat0" to use the Pipecat TTS voice.
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
        persona_id: str = "pipecat0",  # Use `pipecat0` so that your TTS voice is used in place of the Tavus persona
        params: TavusParams = TavusParams(),
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name)
        self._params = params

        # TODO: Filipi - We can remove this if we stop sending the audio through app messages
        # Limiting this so we don't go over 20 messages per second
        # each message is going to have 50ms of audio
        self._params.audio_out_10ms_chunks = 5

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
