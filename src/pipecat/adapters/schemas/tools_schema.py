#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from enum import Enum
from typing import Any, Dict, List, Optional

from pipecat.adapters.schemas.function_schema import FunctionSchema


class AdapterType(Enum):
    GEMINI = "gemini"  # that is the only service where we are able to add custom tools for now


class ToolsSchema:
    def __init__(
        self,
        standard_tools: List[FunctionSchema],
        custom_tools: Optional[Dict[AdapterType, List[Dict[str, Any]]]] = None,
    ) -> None:
        """
        A schema for tools that includes both standardized function schemas
        and custom tools that do not follow the FunctionSchema format.

        :param standard_tools: List of tools following FunctionSchema.
        :param custom_tools: List of tools in a custom format (e.g., search_tool).
        """
        self._standard_tools = standard_tools
        self._custom_tools = custom_tools

    @property
    def standard_tools(self) -> List[FunctionSchema]:
        return self._standard_tools

    @property
    def custom_tools(self) -> Dict[AdapterType, List[Dict[str, Any]]]:
        return self._custom_tools

    @custom_tools.setter
    def custom_tools(self, value: Dict[AdapterType, List[Dict[str, Any]]]) -> None:
        self._custom_tools = value
