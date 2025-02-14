from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from pipecat.services.adapters.function_schema import FunctionSchema


class BaseLLMAdapter(ABC):
    @abstractmethod
    def to_provider_function_format(
        self, functions_schema: Union[FunctionSchema, List[FunctionSchema]]
    ) -> Union[Any, List[Any]]:
        """Converts one or multiple function schemas to the provider's format."""
        pass
