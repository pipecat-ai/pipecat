#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Any, Dict, List, Union

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.tools_schema import ToolsSchema


class GeminiLLMAdapter(BaseLLMAdapter):
    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> List[Dict[str, Any]]:
        """Converts function schemas to Gemini's function-calling format.

        :return: Gemini formatted function call definition.
        """

        functions_schema = tools_schema.standard_tools
        return [{"function_declarations": [func.to_default_dict() for func in functions_schema]}]

        # TODO need to handle google_search and google_search_retrieval
