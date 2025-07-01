#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tools schema definitions for function calling adapters.

This module provides schemas for managing both standardized function tools
and custom adapter-specific tools in the Pipecat framework.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pipecat.adapters.schemas.direct_function import DirectFunction, DirectFunctionWrapper
from pipecat.adapters.schemas.function_schema import FunctionSchema


class AdapterType(Enum):
    """Supported adapter types for custom tools.

    Parameters:
        GEMINI: Google Gemini adapter - currently the only service supporting custom tools.
    """

    GEMINI = "gemini"  # that is the only service where we are able to add custom tools for now


class ToolsSchema:
    """Schema for managing both standard and custom function calling tools.

    This class provides a unified interface for handling standardized function
    schemas alongside custom tools that may not follow the standard format,
    such as adapter-specific search tools.
    """

    def __init__(
        self,
        standard_tools: List[FunctionSchema | DirectFunction],
        custom_tools: Optional[Dict[AdapterType, List[Dict[str, Any]]]] = None,
    ) -> None:
        """Initialize the tools schema.

        Args:
            standard_tools: List of tools following the standardized FunctionSchema format.
            custom_tools: Dictionary mapping adapter types to their custom tool definitions.
                These tools may not follow the FunctionSchema format (e.g., search_tool).
        """

        def _map_standard_tools(tools):
            schemas = []
            for tool in tools:
                if isinstance(tool, FunctionSchema):
                    schemas.append(tool)
                elif callable(tool):
                    wrapper = DirectFunctionWrapper(tool)
                    schemas.append(wrapper.to_function_schema())
                else:
                    raise TypeError(f"Unsupported tool type: {type(tool)}")
            return schemas

        self._standard_tools = _map_standard_tools(standard_tools)
        self._custom_tools = custom_tools

    @property
    def standard_tools(self) -> List[FunctionSchema]:
        """Get the list of standard function schema tools.

        Returns:
            List of tools following the FunctionSchema format.
        """
        return self._standard_tools

    @property
    def custom_tools(self) -> Dict[AdapterType, List[Dict[str, Any]]]:
        """Get the custom tools dictionary.

        Returns:
            Dictionary mapping adapter types to their custom tool definitions.
        """
        return self._custom_tools

    @custom_tools.setter
    def custom_tools(self, value: Dict[AdapterType, List[Dict[str, Any]]]) -> None:
        """Set the custom tools dictionary.

        Args:
            value: Dictionary mapping adapter types to their custom tool definitions.
        """
        self._custom_tools = value
