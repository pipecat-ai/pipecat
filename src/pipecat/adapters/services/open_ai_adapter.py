#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
from typing import List, Union

from openai.types.chat import ChatCompletionToolParam

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.function_schema import FunctionSchema


class OpenAILLMAdapter(BaseLLMAdapter):
    def to_provider_function_format(
        self, functions_schema: Union[FunctionSchema, List[FunctionSchema]]
    ) -> Union[ChatCompletionToolParam, List[ChatCompletionToolParam]]:
        """Converts one or multiple function schemas to OpenAI's function-calling format.

        :return: OpenAI formatted function call definition.
        """
        if isinstance(functions_schema, list):
            # Handling list of FunctionSchema
            return [
                ChatCompletionToolParam(type="function", function=func.to_default_dict())
                for func in functions_schema
            ]
        else:
            # Handling single FunctionSchema
            return ChatCompletionToolParam(
                type="function", function=functions_schema.to_default_dict()
            )
