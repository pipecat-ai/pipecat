from abc import ABC, abstractmethod
from typing import Any, List, Union, cast

from loguru import logger

from pipecat.adapters.schemas.tools_schema import ToolsSchema


class BaseLLMAdapter(ABC):
    @abstractmethod
    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> List[Any]:
        """Converts tools to the provider's format."""
        pass

    def from_standard_tools(self, tools: Any) -> List[Any]:
        if isinstance(tools, ToolsSchema):
            logger.debug(f"Retrieving the tools using the adapter: {type(self)}")
            return self.to_provider_tools_format(tools)
        # Fallback to return the same tools in case they are not in a standard format
        return tools

    # TODO: we can move the logic to also handle the Messages here
