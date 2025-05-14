#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
from dataclasses import dataclass
from typing import Any, Union

from pipecat.frames.frames import (
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    UserImageRawFrame,
)
from pipecat.processors.aggregators.llm_response import (
    BetterLLMUserContextAggregator,
    LLMAssistantAggregatorParams,
    LLMAssistantContextAggregator,
    LLMUserAggregatorParams,
    LLMUserContextAggregator,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.base_llm import BaseOpenAILLMService


@dataclass
class OpenAIContextAggregatorPair:
    _user: Union["OpenAIUserContextAggregator", "PipecatOpenAIUserContextAggregator"]
    _assistant: "OpenAIAssistantContextAggregator"

    def user(self) -> "OpenAIUserContextAggregator":
        return self._user

    def assistant(self) -> "OpenAIAssistantContextAggregator":
        return self._assistant


class OpenAILLMService(BaseOpenAILLMService):
    def __init__(
        self,
        *,
        model: str = "gpt-4.1",
        params: BaseOpenAILLMService.InputParams = BaseOpenAILLMService.InputParams(),
        **kwargs,
    ):
        super().__init__(model=model, params=params, **kwargs)

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
        aggregator_type: str = "ours",  # or "theirs"
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
        if aggregator_type == "ours":
            user = OpenAIUserContextAggregator(context, params=user_params)
        else:
            user = PipecatOpenAIUserContextAggregator(context, params=user_params)
        assistant = OpenAIAssistantContextAggregator(context, params=assistant_params)
        return OpenAIContextAggregatorPair(_user=user, _assistant=assistant)


class OpenAIUserContextAggregator(BetterLLMUserContextAggregator):
    pass


class PipecatOpenAIUserContextAggregator(LLMUserContextAggregator):
    pass


class OpenAIAssistantContextAggregator(LLMAssistantContextAggregator):
    async def handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        self._context.add_message(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": frame.tool_call_id,
                        "function": {
                            "name": frame.function_name,
                            "arguments": json.dumps(frame.arguments),
                        },
                        "type": "function",
                    }
                ],
            }
        )
        self._context.add_message(
            {
                "role": "tool",
                "content": "IN_PROGRESS",
                "tool_call_id": frame.tool_call_id,
            }
        )

    async def handle_function_call_result(self, frame: FunctionCallResultFrame):
        if frame.result:
            result = json.dumps(frame.result)
            await self._update_function_call_result(frame.function_name, frame.tool_call_id, result)
        else:
            await self._update_function_call_result(
                frame.function_name, frame.tool_call_id, "COMPLETED"
            )

    async def handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        await self._update_function_call_result(
            frame.function_name, frame.tool_call_id, "CANCELLED"
        )

    async def _update_function_call_result(
        self, function_name: str, tool_call_id: str, result: Any
    ):
        for message in self._context.messages:
            if (
                message["role"] == "tool"
                and message["tool_call_id"]
                and message["tool_call_id"] == tool_call_id
            ):
                message["content"] = result

    async def handle_user_image_frame(self, frame: UserImageRawFrame):
        await self._update_function_call_result(
            frame.request.function_name, frame.request.tool_call_id, "COMPLETED"
        )
        self._context.add_image_frame_message(
            format=frame.format,
            size=frame.size,
            image=frame.image,
            text=frame.request.context,
        )
