#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AgentHuman API.

API to communicate with AgentHuman Streaming API.
"""

from typing import Any, Dict, Literal, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field

class NewSessionRequest(BaseModel):
    """Requesting model for creating a new AgentHuman session.

    Parameters:
        avatar (str): Unique identifier for the avatar.
        aspect_ratio (Literal["4:3", "3:4", "1:1", "auto"]): Desired aspect ratio of the video stream.
            Use "auto" (default) to let the service infer the best ratio from the transport dimensions.
    """

    avatar: str = "avat_01KMZHXFPBVCXA5ATK85HCP8G1"
    aspect_ratio: Literal["4:3", "3:4", "1:1", "auto"] = "auto"


class AgentHumanRoom(BaseModel):
    """Nested room info within an AgentHuman session.

    Parameters:
        roomURL (str): WebSocket URL of the LiveKit room.
        participantToken (str): Token for joining the LiveKit room.
    """

    roomURL: str
    participantToken: str


class AgentHumanSession(BaseModel):
    """Response model for a AgentHuman session.

    Parameters:
        session_id (str): Unique identifier for the streaming session.
        session_token (str): Token for accessing the session securely.
        started_at (Optional[str]): ISO timestamp when the session started.
        ended_at (Optional[str]): ISO timestamp when the session ended.
        expiration (Optional[str]): ISO timestamp when the session expires.
        aspect_ratio (str): Aspect ratio of the video stream.
        duration (int): Duration of the session in seconds.
        status (str): Status of the session.
        metadata (Dict[str, Any]): Arbitrary metadata attached to the session.
        room (AgentHumanRoom): LiveKit room connection details.
    """

    session_id: str
    session_token: str
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    expiration: Optional[str] = None
    aspect_ratio: str = "auto"
    duration: int = 0
    status: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    room: AgentHumanRoom

    @property
    def roomURL(self) -> str:
        return self.room.roomURL

    @property
    def participantToken(self) -> str:
        return self.room.participantToken


class AgentHumanApiError(Exception):
    """Custom exception for AgentHuman API errors."""

    def __init__(self, message: str, status: int, response_text: str) -> None:
        """Initialize the AgentHuman API error.

        Args:
            message: Error message
            status: HTTP status code
            response_text: Raw response text from the API
        """
        super().__init__(message)
        self.status = status
        self.response_text = response_text


class AgentHumanApi:
    """AgentHuman Streaming API client."""

    BASE_URL = "https://api.agenthuman.com/v1"

    def __init__(self, api_key: str) -> None:
        """Initialize the AgentHuman API.

        Args:
            api_key: AgentHuman API key
        """
        self.api_key = api_key

    async def _request(self, path: str, params: Dict[str, Any], expect_data: bool = True) -> Any:
        """Make a POST request to the AgentHuman API.

        Args:
            path: API endpoint path.
            params: JSON-serializable parameters.
            expect_data: Whether to expect and extract 'data' field from response (default: True).

        Returns:
            Parsed JSON response data.

        Raises:
            AgentHumanApiError: If the API response is not successful or data is missing when expected.
            aiohttp.ClientError: For network-related errors.
        """
        url = f"{self.BASE_URL}{path}"
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        logger.debug(f"AgentHuman API request: {url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=params, headers=headers) as response:
                    if not response.ok:
                        response_text = await response.text()
                        logger.error(f"AgentHuman API error: {response_text}")
                        raise AgentHumanApiError(
                            f"API request failed with status {response.status}",
                            response.status,
                            response_text,
                        )
                    if expect_data:
                        json_data = await response.json()
                        return json_data
                    return await response.text()
        except aiohttp.ClientError as e:
            logger.error(f"Network error while calling AgentHuman API: {str(e)}")
            raise

    async def new_session(self, request_data: NewSessionRequest) -> AgentHumanSession:
        """Create a new streaming session.

        https://docs.agenthuman.com/api-reference/endpoints/create-session

        Args:
            request_data: Session configuration parameters.

        Returns:
            Session information, including ID and access token.
        """
        params = {
            "avatar": request_data.avatar,
            "aspect_ratio": request_data.aspect_ratio,
            "room": {
                "platform": "proxy"
            }
        }
        session_info = await self._request("/sessions", params)
        session_info = session_info.get("session")

        return AgentHumanSession.model_validate(session_info)

    async def end_session(self, session_id: str) -> Any:
        """Terminate an active the streaming session.

        https://docs.agenthuman.com/api-reference/endpoints/end-session

        Args:
            session_id: ID of the session to stop.

        Returns:
            Response data from the stop session API call.

        Raises:
            ValueError: If session ID is not set.
        """
        if not session_id:
            raise ValueError("Session ID is not set. Call new_session first.")

        return await self._request(f"/sessions/{session_id}/end", {}, expect_data=False)