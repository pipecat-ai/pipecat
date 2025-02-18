#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Any, Dict, List

from pipecat.adapters.schemas.function_schema import FunctionSchema


class ToolsSchema:
    def __init__(
        self, standard_tools: List[FunctionSchema], custom_tools: List[Dict[str, Any]] = None
    ) -> None:
        """
        A schema for tools that includes both standardized function schemas
        and custom tools that do not follow the FunctionSchema format.

        :param standard_tools: List of tools following FunctionSchema.
        :param custom_tools: List of tools in a custom format (e.g., search_tool).
        """
        self.standard_tools = standard_tools
        self.custom_tools = custom_tools
