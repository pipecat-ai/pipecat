"""Hume Speech-to-Speech service using raw WebSocket connection.

This implementation connects directly to Hume EVI WebSocket API without using the SDK,
based on the orchestrator.py approach but following pipecat service patterns.
"""

import asyncio
import base64
import io
import wave
from dataclasses import dataclass

from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AggregationType,
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
from pipecat.services.hume.hume_client import SimpleHumeClient
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
    """Frame to trigger Hume connection start."""

    pass


class HumeSTSService(LLMService):
    """Hume Speech-to-Speech service using raw WebSocket connection.

    This service connects directly to Hume's EVI WebSocket API without using the SDK.
    It provides more control over the connection and message handling.
    """

    def __init__(
        self,
        *,
        api_key: str,
        config_id: str,
        model: str = "evi",
        system_prompt: str | None = None,
        start_frame_cls: type[Frame] | None = None,
        track_cancelled_conversations: bool = True,
        audio_passthrough: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        logger.debug("Initializing HumeRawSTSService")
        self._audio_passthrough = audio_passthrough
        self.api_key = api_key
        self.config_id = config_id
        self.model = model
        self.system_prompt = system_prompt
        self._client: SimpleHumeClient | None = None
        self.active_conversation: bool = False
        self.active_conversation_id: str | None = None
        self.track_cancelled_conversations = track_cancelled_conversations
        self.cancelled_conversation_ids: list[str] = []
        self._context: OpenAIRealtimeLLMContext | None = None
        self._resampler = create_stream_resampler()
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
            logger.info("Starting Hume Raw service")
            await self._connect()
            await self.push_frame(frame, direction)

        elif isinstance(frame, StartInterruptionFrame):
            if self.active_conversation_id is not None and self.track_cancelled_conversations:
                self.cancelled_conversation_ids.append(self.active_conversation_id)
            self.active_conversation_id = None
            self.active_conversation = False
            await self.push_frame(frame)

        elif isinstance(frame, InputAudioRawFrame):
            if self._client and self._client.is_connected:
                audio = await self._resampler.resample(frame.audio, frame.sample_rate, 16000)
                await self._client.send_audio(audio)
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
                self._context = context
                await self.reset_conversation()
        else:
            await self.push_frame(frame, direction)

    async def _connect(self):
        """Connect to Hume EVI WebSocket API."""
        try:
            logger.info("Connecting to Hume EVI (SimpleHumeClient)...")
            self._client = SimpleHumeClient(
                on_message=lambda data: asyncio.create_task(self._handle_message(data)),
                api_key=self.api_key,
                config_id=self.config_id,
                system_prompt=self.system_prompt,
            )
            await self._client.connect()
        except Exception as e:
            logger.error(f"Failed to connect to Hume EVI: {e}")
            await self.push_frame(ErrorFrame(error=str(e), fatal=True))

    async def _disconnect(self):
        """Disconnect from Hume EVI."""
        if self._client:
            await self._client.disconnect()
            self._client = None
            logger.info("Disconnected from Hume EVI")

    async def _handle_message(self, data: dict):
        """Handle incoming message from Hume EVI."""
        msg_type = data.get("type")
        msg_id = data.get("id")

        if msg_id and msg_id in self.cancelled_conversation_ids and self.active_conversation:
            logger.debug(f"Skipping message from cancelled conversation {msg_id}")
            return

        if msg_type == "audio_output":
            await self._handle_audio_output(data)

        elif msg_type == "assistant_end":
            logger.info(f"Assistant end for conversation id: {self.active_conversation_id}")
            self.active_conversation = False
            await self.push_frame(LLMFullResponseEndFrame())
            await self.push_frame(TTSStoppedFrame())

        elif msg_type == "assistant_message":
            message = data.get("message", {})
            content = message.get("content", "")
            if content:
                logger.info(f"Assistant message: {content[:50]}...")
                await self.push_frame(LLMTextFrame(text=content))
                await self.push_frame(
                    TTSTextFrame(text=content, aggregated_by=AggregationType.SENTENCE)
                )

        elif msg_type == "user_message":
            message = data.get("message", {})
            content = message.get("content", "")
            is_interim = data.get("interim", False)
            if content:
                logger.info(f"User message (interim={is_interim}): {content[:50]}...")
                await self.push_frame(
                    TranscriptionFrame(
                        text=content,
                        user_id="",
                        timestamp=time_now_iso8601(),
                        result=data,
                    )
                )

        elif msg_type == "user_interruption":
            logger.info("User interruption received")
            await self.push_frame(StartInterruptionFrame())

        elif msg_type == "chat_metadata":
            logger.info(f"Hume chat metadata: {data}")

        elif msg_type == "error":
            error_msg = data.get("message", "Unknown error")
            logger.error(f"Hume error: {error_msg}")
            await self.push_frame(ErrorFrame(error=error_msg, fatal=False))

    async def _handle_audio_output(self, data: dict):
        """Handle audio_output message from Hume EVI."""
        self.active_conversation_id = data.get("id")

        if not self.active_conversation:
            self.active_conversation = True
            await self.push_frame(TTSStartedFrame())
            await self.push_frame(LLMFullResponseStartFrame())

        wav_b64 = data.get("data", "")
        if not wav_b64:
            return

        wav_bytes = base64.b64decode(wav_b64)

        try:
            with io.BytesIO(wav_bytes) as wav_file:
                with wave.open(wav_file, "rb") as wav_reader:
                    sample_rate = wav_reader.getframerate()
                    num_channels = wav_reader.getnchannels()
                    audio_frames = wav_reader.readframes(wav_reader.getnframes())

            frame = TTSAudioRawFrame(
                audio=audio_frames, sample_rate=sample_rate, num_channels=num_channels
            )
            await self.push_frame(frame)

            samples_count = len(audio_frames) / 2
            duration = samples_count / num_channels / sample_rate
            logger.info(
                f"Received audio from Hume id: {self.active_conversation_id}, "
                f"samples: {samples_count}, channels: {num_channels}, duration: {duration:.2f}s"
            )

        except Exception as e:
            logger.error(f"Failed to decode WAV audio: {e}")

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
