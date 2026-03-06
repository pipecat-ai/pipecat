"""Strategy interfaces for call operations in serializers.

This module defines the abstract interfaces that telephony serializers can use
to delegate call operations (transfer, hangup) to provider-specific implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from loguru import logger


class CallOperationStrategy(ABC):
    """Base strategy for call operations."""

    pass


class TransferStrategy(CallOperationStrategy):
    """Strategy for handling call transfer operations.

    Implementations should handle all aspects of transferring a call.
    """

    @abstractmethod
    async def execute_transfer(self, context: Dict[str, Any]) -> bool:
        """Execute call transfer with provider-specific logic.

        Args:
            context: Dictionary containing all necessary transfer context:
                - Provider-specific connection details
                - Call identifiers
                - Transfer destination information
                - Any other context needed for the operation

        Returns:
            bool: True if transfer was successful, False otherwise
        """
        pass


class HangupStrategy(CallOperationStrategy):
    """Strategy for handling call hangup operations."""

    @abstractmethod
    async def execute_hangup(self, context: Dict[str, Any]) -> bool:
        """Execute call hangup with provider-specific logic.

        Args:
            context: Dictionary containing all necessary hangup context:
                - Provider-specific connection details
                - Call identifiers
                - Any other context needed for the operation

        Returns:
            bool: True if hangup was successful, False otherwise
        """
        pass
