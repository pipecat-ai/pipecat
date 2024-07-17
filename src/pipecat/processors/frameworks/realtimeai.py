#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import dataclasses

from typing import List, Literal, Optional, Type
from pydantic import BaseModel, ValidationError

from pipecat.frames.frames import Frame, StartFrame, TransportMessageFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_response import LLMAssistantResponseAggregator, LLMUserResponseAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.ai_services import AIService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.ollama import OLLamaLLMService
from pipecat.transports.base_transport import BaseTransport
from pipecat.vad.silero import SileroVAD


class RealtimeAILLMConfig(BaseModel):
    model: str
    messages: List[dict]


class RealtimeAITTSConfig(BaseModel):
    voice: str


class RealtimeAIConfig(BaseModel):
    llm: RealtimeAILLMConfig
    tts: RealtimeAITTSConfig


class RealtimeAIMessageData(BaseModel):
    config: Optional[RealtimeAIConfig] = None


class RealtimeAIMessage(BaseModel):
    tag: Literal["realtime-ai"] = "realtime-ai"
    type: str
    data: RealtimeAIMessageData


class RealtimeAIResponseMessage(BaseModel):
    tag: Literal["realtime-ai"] = "realtime-ai"
    type: str
    success: bool
    error: Optional[str] = None


class RealtimeAIProcessor(FrameProcessor):

    def __init__(
            self,
            *,
            transport: BaseTransport,
            config: RealtimeAIConfig | None = None,
            llm_api_key: str = "",
            tts_api_key: str = "",
            llm_cls: Type[AIService] = OLLamaLLMService,
            tts_cls: Type[AIService] = CartesiaTTSService):
        super().__init__()
        self._transport = transport
        self._config = config
        self._llm_api_key = llm_api_key
        self._tts_api_key = tts_api_key
        self._llm_cls = llm_cls
        self._tts_cls = tts_cls
        self._start_frame: Frame | None = None
        self._llm: FrameProcessor | None = None
        self._tts: FrameProcessor | None = None
        self._pipeline: FrameProcessor | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, TransportMessageFrame):
            await self._handle_message(frame)
        else:
            await self.push_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._start_frame = frame
            if self._config:
                await self._handle_config(self._config)

    async def _handle_message(self, frame: TransportMessageFrame):
        try:
            message = RealtimeAIMessage.model_validate(frame.message)

            match message.type:
                case "config":
                    await self._handle_config(RealtimeAIConfig.model_validate(message.data.config))
        except ValidationError as e:
            await self._send_response("config", False, f"invalid configuration: {e}")

    async def _handle_config(self, config: RealtimeAIConfig):
        try:
            tma_in = LLMUserResponseAggregator(config.llm.messages)
            tma_out = LLMAssistantResponseAggregator(config.llm.messages)

            vad = SileroVAD()

            self._llm = self._llm_cls(model=config.llm.model)

            self._tts = self._tts_cls(api_key=self._tts_api_key, voice_id=config.tts.voice)

            pipeline = Pipeline([vad, tma_in, self._llm, self._tts,
                                self._transport.output(), tma_out])
            self._pipeline = pipeline

            parent = self.get_parent()
            if parent and self._start_frame:
                parent.link(pipeline)

                # We need to initialize the new pipeline with the same settings
                # as the initial one.
                start_frame = dataclasses.replace(self._start_frame)
                await self.push_frame(start_frame)

                # We now send a message to indicate we successfully initialized
                # the pipelines.
                await self._send_response("config", True)
        except Exception as e:
            await self._send_response("config", False, f"unable to create pipeline: {e}")

    async def _send_response(self, type: str, success: bool, error: str | None = None):
        response = RealtimeAIResponseMessage(type=type, success=success)
        message = TransportMessageFrame(message=response.model_dump(exclude_none=True))
        await self.push_frame(message)
