from abc import ABC, abstractmethod
from typing import Any, List, Union, cast

from loguru import logger

from pipecat.adapters.function_schema import FunctionSchema


class BaseLLMAdapter(ABC):
    @abstractmethod
    def to_provider_function_format(
        self, functions_schema: Union[FunctionSchema, List[FunctionSchema]]
    ) -> Union[Any, List[Any]]:
        """Converts one or multiple function schemas to the provider's format."""
        pass

    def from_standard_tools(self, tools: Any) -> List[Any]:
        if isinstance(tools, list) and all(isinstance(tool, FunctionSchema) for tool in tools):
            logger.debug(f"Retrieving the tools using the adapter: {type(self)}")
            return self.to_provider_function_format(cast(List[FunctionSchema], tools))
        # Fallback to return the same tools in case they are not in a standard format
        return tools

    # TODO: we can move the logic to also adapter the Messages to here
