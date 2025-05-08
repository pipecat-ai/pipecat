#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import List, Literal, Optional

from pydantic import BaseModel

from pipecat.frames.frames import Frame
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.services.google.frames import LLMSearchOrigin, LLMSearchResponseFrame


class RTVISearchResponseMessageData(BaseModel):
    search_result: Optional[str]
    rendered_content: Optional[str]
    origins: List[LLMSearchOrigin]


class RTVIBotLLMSearchResponseMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["bot-llm-search-response"] = "bot-llm-search-response"
    data: RTVISearchResponseMessageData


class GoogleRTVIObserver(RTVIObserver):
    def __init__(self, rtvi: RTVIProcessor):
        super().__init__(rtvi)

    async def on_push_frame(self, data: FramePushed):
        await super().on_push_frame(data)

        frame = data.frame

        if isinstance(frame, LLMSearchResponseFrame):
            await self._handle_llm_search_response_frame(frame)

    async def _handle_llm_search_response_frame(self, frame: LLMSearchResponseFrame):
        message = RTVIBotLLMSearchResponseMessage(
            data=RTVISearchResponseMessageData(
                search_result=frame.search_result,
                origins=frame.origins,
                rendered_content=frame.rendered_content,
            )
        )
        await self.push_transport_message_urgent(message)
