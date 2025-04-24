#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
from abc import ABC, abstractmethod
from dataclasses import field
from typing import Literal, Optional

import httpx
from dotenv import load_dotenv
from loguru import logger
from openai import BaseModel

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.ai_service import AIService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import (
    OpenAIAssistantContextAggregator,
    OpenAIContextAggregatorPair,
    OpenAILLMService,
    OpenAIUserContextAggregator,
)
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection

load_dotenv(override=True)


class LlmMessage(BaseModel):
    #     ...
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str]


class AgentResponse(BaseModel):
    content: str
    msgs: list[LlmMessage] = field(default_factory=list)


class BackendBase(ABC):
    @abstractmethod
    async def get_resp(self, messages: list[LlmMessage], extra_params) -> AgentResponse:
        raise NotImplementedError("The method get_resp is not implemented.")


class CompassLLMService(AIService):
    def __init__(self, backend: BackendBase):
        super().__init__()
        self.backend = backend

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> OpenAIContextAggregatorPair:
        """Create an instance of OpenAIContextAggregatorPair from an
        OpenAILLMContext. Constructor keyword arguments for both the user and
        assistant aggregators can be provided.

        Args:
            context (OpenAILLMContext): The LLM context.
            user_params (LLMUserAggregatorParams, optional): User aggregator parameters.
            assistant_params (LLMAssistantAggregatorParams, optional): User aggregator parameters.

        Returns:
            OpenAIContextAggregatorPair: A pair of context aggregators, one for
            the user and one for the assistant, encapsulated in an
            OpenAIContextAggregatorPair.

        """
        context.set_llm_adapter(self.get_llm_adapter())
        user = OpenAIUserContextAggregator(context, params=user_params)
        assistant = OpenAIAssistantContextAggregator(context, params=assistant_params)
        return OpenAIContextAggregatorPair(_user=user, _assistant=assistant)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, OpenAILLMContextFrame):
            context: OpenAILLMContext = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = OpenAILLMContext.from_messages(frame.messages)
        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame.settings)
        else:
            await self.push_frame(frame, direction)

        if context:
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.start_processing_metrics()
                # await self._process_context(context)

                msgs = []
                for contmsg in context.messages:
                    msgs.append(
                        LlmMessage(
                            role=contmsg["role"],
                            content=contmsg["content"],
                        )
                    )
                resp = await self.backend.get_resp(
                    msgs,
                    {
                        "conversation_id": "fake_conversation_id",
                        "user_id": "fake_user_id",
                    },
                )

                context.add_messages(resp.msgs)
                await self.push_frame(LLMTextFrame(resp.content))

            except httpx.TimeoutException:
                await self._call_event_handler("on_completion_timeout")
            finally:
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())


async def run_bot(webrtc_connection: SmallWebRTCConnection):
    logger.info(f"Starting bot")

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        ),
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages=messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            report_only_initial_ttfb=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")

    @transport.event_handler("on_client_closed")
    async def on_client_closed(transport, client):
        logger.info(f"Client closed connection")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


if __name__ == "__main__":
    from run import main

    main()
