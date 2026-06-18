#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tools schema definitions for function calling adapters.

This module provides schemas for managing both standardized function tools
and custom adapter-specific tools in the Pipecat framework.
"""

from collections.abc import Sequence
from enum import Enum
from typing import Any

from pipecat.adapters.schemas.direct_function import DirectFunction, DirectFunctionWrapper
from pipecat.adapters.schemas.function_schema import FunctionSchema


class AdapterType(Enum):
    """Supported adapter types for custom tools.

    Parameters:
        GEMINI: Google Gemini adapter.
        OPENAI: OpenAI adapter (Chat Completions, Responses, and Realtime API).
    """

    GEMINI = "gemini"
    OPENAI = "openai"


class ToolsSchema:
    """Schema for managing both standard and custom function calling tools.

    This class provides a unified interface for handling standardized function
    schemas alongside custom tools that may not follow the standard format,
    such as adapter-specific search tools.
    """

    def __init__(
        self,
        standard_tools: Sequence[FunctionSchema | DirectFunction],
        custom_tools: dict[AdapterType, list[dict[str, Any]]] | None = None,
    ) -> None:
        """Initialize the tools schema.

        Args:
            standard_tools: List of tools following the standardized FunctionSchema format.
            custom_tools: Dictionary mapping adapter types to their custom tool definitions.
                These tools may not follow the FunctionSchema format (e.g., search_tool).
        """

        def _map_standard_tools(tools):
            schemas = []
            direct_functions = []
            for tool in tools:
                if isinstance(tool, FunctionSchema):
                    schemas.append(tool)
                elif callable(tool):
                    wrapper = DirectFunctionWrapper(tool)
                    schemas.append(wrapper.to_function_schema())
                    direct_functions.append(wrapper)
                else:
                    raise TypeError(f"Unsupported tool type: {type(tool)}")
            return schemas, direct_functions

        self._standard_tools, self._direct_functions = _map_standard_tools(standard_tools)
        self._custom_tools = custom_tools

    @property
    def standard_tools(self) -> list[FunctionSchema]:
        """Get the list of standard function schema tools.

        Returns:
            List of tools following the FunctionSchema format.
        """
        return self._standard_tools

    @property
    def direct_functions(self) -> list[DirectFunctionWrapper]:
        """Get the wrappers for standard tools that were given as direct functions.

        These retain a reference to the original callable, allowing the LLM
        service to register their handlers automatically. Standard tools given
        as ``FunctionSchema`` objects (advertise-only) are not included.

        Returns:
            List of ``DirectFunctionWrapper`` for the callable standard tools.
        """
        return self._direct_functions

    @property
    def custom_tools(self) -> dict[AdapterType, list[dict[str, Any]]] | None:
        """Get the custom tools dictionary.

        Returns:
            Dictionary mapping adapter types to their custom tool definitions.
        """
        return self._custom_tools

    @custom_tools.setter
    def custom_tools(self, value: dict[AdapterType, list[dict[str, Any]]]) -> None:
        """Set the custom tools dictionary.

        Args:
            value: Dictionary mapping adapter types to their custom tool definitions.
        """
        self._custom_tools = value
