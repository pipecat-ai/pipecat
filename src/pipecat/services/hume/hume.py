"""Hume Speech-to-Speech (STS) service based on LLMService."""

import base64
import io
import time
import wave
from dataclasses import dataclass

from hume import AsyncHumeClient
from hume.empathic_voice import (
    AudioConfiguration,
    AudioInput,
    Context,
    SessionSettings,
    SubscribeEvent,
)
from hume.empathic_voice.chat.socket_client import (
    ChatConnectOptions,
    ChatWebsocketConnection,
)
from loguru import logger
from websockets import ConnectionClosed

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    StartInterruptionFrame,
    SystemFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.llm import OpenAIContextAggregatorPair
from pipecat.services.openai_realtime_beta.context import (
    OpenAIRealtimeAssistantContextAggregator,
    OpenAIRealtimeLLMContext,
    OpenAIRealtimeUserContextAggregator,
)
from pipecat.utils.time import time_now_iso8601


@dataclass
class HumeStartFrame(SystemFrame):
    pass


class HumeSTSService(LLMService):
    """Hume Speech-to-Speech service.

    This service uses Hume's Empathic Voice Interface (EVI) to provide speech-to-speech
    functionality. It sends audio frames to Hume and receives audio frames in response.
    """

    def __init__(
        self,
        *,
        api_key: str,
        config_id: str,
        model: str = "evi",
        system_prompt: str | None = None,
        start_frame_cls: type[Frame] | None = None,
        audio_passthrough: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        logger.debug("Initializing HumeSTSService")
        self._audio_passthrough = audio_passthrough
        self.api_key = api_key
        self.config_id = config_id
        self.model = model
        self._client = AsyncHumeClient(api_key=self.api_key)
        self._connection: ChatWebsocketConnection | None = None
        self._cm = None
        self.active_conversation: bool = False
        self.active_conversation_id: str | None = None
        self.cancelled_conversation_ids: list[str] = []
        self.system_prompt = system_prompt
        self._context: OpenAIRealtimeLLMContext | None = None
        self._hume_context: Context | None = None
        self._time_to_first_audio_list = []
        self._start_frame_cls = start_frame_cls or HumeStartFrame

    async def start(self, frame: StartFrame):
        await super().start(frame)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def reset_conversation(self):
        await self._disconnect()
        await self._connect()
        self.active_conversation = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, self._start_frame_cls):
            await self._connect()
        elif isinstance(frame, StartInterruptionFrame):
            if self.active_conversation_id is not None:
                self.cancelled_conversation_ids.append(self.active_conversation_id)
            self.active_conversation_id = None
            self.active_conversation = False
            await self.push_frame(frame)

        elif isinstance(frame, InputAudioRawFrame):
            if self._connection:
                encoded_audio = base64.b64encode(frame.audio).decode("utf-8")
                input = AudioInput(data=encoded_audio)
                while True:
                    try:
                        await self._connection.send_audio_input(input)
                        break
                    except ConnectionClosed:
                        await self.reset_conversation()
            if self._audio_passthrough:
                await self.push_frame(frame, direction)

        elif isinstance(frame, OpenAILLMContextFrame):
            logger.info("OpenAILLMContextFrame frame received")
            context: OpenAIRealtimeLLMContext = OpenAIRealtimeLLMContext.upgrade_to_realtime(
                frame.context
            )
            if not self._context:
                self._context = context
            elif frame.context is not self._context:
                # If the context has changed, reset the conversation
                self._context = context
                await self.reset_conversation()
        else:
            await self.push_frame(frame, direction)

    async def _connect(self):
        try:
            self._cm = self._client.empathic_voice.chat.connect_with_callbacks(
                options=ChatConnectOptions(config_id=self.config_id),
                on_message=self._on_message,
                on_open=self._on_open,
                on_close=self._on_close,
                on_error=self._on_error,
            )
            self._connection = await self._cm.__aenter__()
            await self._connection.send_session_settings(
                SessionSettings(
                    type="session_settings",
                    system_prompt=self.system_prompt,
                    audio=AudioConfiguration(
                        encoding="linear16",
                        sample_rate=16000,
                        channels=1,
                    ),
                    context=self._hume_context,
                )
            )
        except Exception as e:
            logger.error(f"Failed to connect to Hume EVI: {e}")
            await self.push_error(ErrorFrame(error=str(e), fatal=True))

    async def _disconnect(self):
        if self._cm:
            await self._cm.__aexit__(None, None, None)
            self._cm = None
            self._connection = None

    async def _on_open(self):
        logger.info("Connected to Hume EVI")

    async def _on_close(self):
        logger.info("Disconnected from Hume EVI")

    async def _on_error(self, error: Exception):
        logger.error(f"Hume EVI error: {error}")
        await self.push_error(ErrorFrame(error=str(error), fatal=True))

    async def _on_message(self, message: SubscribeEvent):
        logger.trace(f"Received message from Hume: {message}")
        msg_type = message.type
        if hasattr(message, "id") and message.id in self.cancelled_conversation_ids:
            return

        if msg_type == "audio_output":
            self.active_conversation_id = message.id
            if not self.active_conversation:
                self.active_conversation = True
                await self.push_frame(TTSStartedFrame())
                await self.push_frame(LLMFullResponseStartFrame())

            wav_data = base64.b64decode(message.data.encode("utf-8"))
            with io.BytesIO(wav_data) as wav_file:
                with wave.open(wav_file, "rb") as wav_reader:
                    sample_rate = wav_reader.getframerate()
                    num_channels = wav_reader.getnchannels()
                    audio_frames = wav_reader.readframes(wav_reader.getnframes())

            frame = TTSAudioRawFrame(
                audio=audio_frames, sample_rate=sample_rate, num_channels=num_channels
            )
            await self.push_frame(frame)
            samples_count = len(audio_frames) / 2
            logger.info(
                f"Received audio samples from HumeAI id: {message.id}, {samples_count} samples, channels: {num_channels}, duration: {samples_count / num_channels / sample_rate}"
            )

        elif msg_type == "assistant_end":
            self.active_conversation = False
            await self.push_frame(LLMFullResponseEndFrame())
            await self.push_frame(TTSStoppedFrame())
        elif msg_type == "assistant_message":
            logger.info(message)
            content: str = message.message.content
            await self.push_frame(LLMTextFrame(text=content))
            await self.push_frame(TTSTextFrame(text=content))
        elif msg_type == "user_message":
            content: str = message.message.content
            logger.info(message)
            await self.push_frame(
                TranscriptionFrame(
                    text=content, user_id="", timestamp=time_now_iso8601(), result=message
                )
            )
        elif msg_type == "chat_metadata":
            logger.info(f"Hume chat metadata: {message}")
        elif msg_type == "error":
            await self.push_error(ErrorFrame(error=message.message, fatal=False))

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> OpenAIContextAggregatorPair:
        """Create an instance of OpenAIContextAggregatorPair from an OpenAILLMContext.

        Constructor keyword arguments for both the user and assistant aggregators can be provided.

        Args:
            context: The LLM context.
            user_params: User aggregator parameters.
            assistant_params: Assistant aggregator parameters.

        Returns:
            OpenAIContextAggregatorPair: A pair of context aggregators, one for
            the user and one for the assistant, encapsulated in an
            OpenAIContextAggregatorPair.
        """
        context.set_llm_adapter(self.get_llm_adapter())

        OpenAIRealtimeLLMContext.upgrade_to_realtime(context)
        user = OpenAIRealtimeUserContextAggregator(context, params=user_params)

        assistant_params.expect_stripped_words = False
        assistant = OpenAIRealtimeAssistantContextAggregator(context, params=assistant_params)
        return OpenAIContextAggregatorPair(_user=user, _assistant=assistant)
