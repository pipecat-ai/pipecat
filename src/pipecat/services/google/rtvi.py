#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Google RTVI integration models and observer implementation.

This module provides integration with Google's services through the RTVI framework,
including models for search responses and an observer for handling Google-specific
frame types.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel

from pipecat.frames.frames import Frame
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.services.google.frames import LLMSearchOrigin, LLMSearchResponseFrame


class RTVISearchResponseMessageData(BaseModel):
    """Data payload for search response messages in RTVI protocol.

    Parameters:
        search_result: The search result text, if available.
        rendered_content: The rendered content from the search, if available.
        origins: List of search result origins with metadata.
    """

    search_result: Optional[str]
    rendered_content: Optional[str]
    origins: List[LLMSearchOrigin]


class RTVIBotLLMSearchResponseMessage(BaseModel):
    """RTVI message for bot LLM search responses.

    Parameters:
        label: Always "rtvi-ai" for RTVI protocol messages.
        type: Always "bot-llm-search-response" for this message type.
        data: The search response data payload.
    """

    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["bot-llm-search-response"] = "bot-llm-search-response"
    data: RTVISearchResponseMessageData


class GoogleRTVIObserver(RTVIObserver):
    """RTVI observer for Google service integration.

    Extends the base RTVIObserver to handle Google-specific frame types,
    particularly LLM search response frames from Google services.
    """

    def __init__(self, rtvi: RTVIProcessor):
        """Initialize the Google RTVI observer.

        Args:
            rtvi: The RTVI processor to send messages through.
        """
        super().__init__(rtvi)

    async def on_push_frame(self, data: FramePushed):
        """Process frames being pushed through the pipeline.

        Handles Google-specific frames in addition to the base RTVI frame types.

        Args:
            data: Frame push event data containing frame and metadata.
        """
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
