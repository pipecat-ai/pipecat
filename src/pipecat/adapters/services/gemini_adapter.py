#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Any, Dict, List, Union

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.function_schema import FunctionSchema


# TODO need to think about how we are going to handle the Google Search tools
# google_search and google_search_retrieval
# Look at news_bot and 26e-gemini-multimodal-google-search for more details
class GeminiLLMAdapter(BaseLLMAdapter):
    def to_provider_function_format(
        self, functions_schema: Union[FunctionSchema, List[FunctionSchema]]
    ) -> Dict[str, Any]:
        """Converts one or multiple function schemas to Gemini's function-calling format.

        :return: Gemini formatted function call definition.
        """
        if isinstance(functions_schema, list):
            # Handling list of FunctionSchema
            return [
                {"function_declarations": [func.to_default_dict() for func in functions_schema]}
            ]
        else:
            # Handling single FunctionSchema
            print(f"{functions_schema}")
            return functions_schema.to_default_dict()
