#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base API for HeyGen avatar services.

Base class defining the common interface for HeyGen avatar service APIs.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class StandardSessionResponse(BaseModel):
    """Standardized session response that all HeyGen avatar services will provide.

    This contains the common fields that the client needs to operate,
    while also storing the raw response for service-specific data access.

    Parameters:
        session_id (str): Unique identifier for the streaming session.
        access_token (str): Token for accessing the session securely.
        livekit_agent_token (str): Token for HeyGen’s audio agents(Pipecat).
        ws_url (str): WebSocket URL for the session.
        livekit_url (str): LiveKit server URL for the session.
    """

    session_id: str
    access_token: str
    livekit_agent_token: str

    livekit_url: str = None
    ws_url: str = None

    raw_response: Any


class BaseAvatarApi(ABC):
    """Base class for avatar service APIs."""

    @abstractmethod
    async def new_session(self, request_data: Any) -> StandardSessionResponse:
        """Create a new avatar session.

        Args:
            request_data: Service-specific session request data

        Returns:
            StandardSessionResponse: Standardized session information
        """
        pass

    @abstractmethod
    async def close_session(self, session_id: str) -> Any:
        """Close an avatar session.

        Args:
            session_id: ID of the session to close

        Returns:
            Response data from the close session API call
        """
        pass
