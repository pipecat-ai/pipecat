#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
from typing import List

from openai.types.chat import ChatCompletionToolParam

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.tools_schema import ToolsSchema


class OpenAILLMAdapter(BaseLLMAdapter):
    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> List[ChatCompletionToolParam]:
        """Converts function schemas to OpenAI's function-calling format.

        :return: OpenAI formatted function call definition.
        """
        functions_schema = tools_schema.standard_tools
        return [
            ChatCompletionToolParam(type="function", function=func.to_default_dict())
            for func in functions_schema
        ]
