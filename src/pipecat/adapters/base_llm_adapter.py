#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base adapter for LLM provider integration.

This module provides the abstract base class for implementing LLM provider-specific
adapters that handle tool format conversion and standardization.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Union, cast

from loguru import logger

from pipecat.adapters.schemas.tools_schema import ToolsSchema


class BaseLLMAdapter(ABC):
    """Abstract base class for LLM provider adapters.

    Provides a standard interface for converting between Pipecat's standardized
    tool schemas and provider-specific tool formats. Subclasses must implement
    provider-specific conversion logic.
    """

    @abstractmethod
    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> List[Any]:
        """Convert tools schema to the provider's specific format.

        Args:
            tools_schema: The standardized tools schema to convert.

        Returns:
            List of tools in the provider's expected format.
        """
        pass

    def from_standard_tools(self, tools: Any) -> List[Any]:
        """Convert tools from standard format to provider format.

        Args:
            tools: Tools in standard format or provider-specific format.

        Returns:
            List of tools converted to provider format, or original tools
            if not in standard format.
        """
        if isinstance(tools, ToolsSchema):
            logger.debug(f"Retrieving the tools using the adapter: {type(self)}")
            return self.to_provider_tools_format(tools)
        # Fallback to return the same tools in case they are not in a standard format
        return tools

    # TODO: we can move the logic to also handle the Messages here
