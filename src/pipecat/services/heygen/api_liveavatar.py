#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LiveAvatar API.

API to communicate with LiveAvatar Streaming API.
"""

from typing import Any, Dict, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.services.heygen.base_api import BaseAvatarApi, StandardSessionResponse


class AvatarPersona(BaseModel):
    """Avatar persona settings for LiveAvatar.

    Parameters:
        voice_id (Optional[str]): ID of the voice to be used.
        context_id (Optional[str]): Context ID for the avatar.
        language (str): Language code for the avatar (default: "en").
    """

    voice_id: Optional[str] = None
    context_id: Optional[str] = None
    language: str = "en"


class CustomSDKLiveKitConfig(BaseModel):
    """Custom LiveKit configuration.

    Parameters:
        livekit_url (str): LiveKit server URL.
        livekit_room (str): LiveKit room name.
        livekit_client_token (str): LiveKit client access token.
    """

    livekit_url: str
    livekit_room: str
    livekit_client_token: str


class LiveAvatarNewSessionRequest(BaseModel):
    """Request model for creating a LiveAvatar session token.

    Parameters:
        mode (str): Session mode (default: "CUSTOM").
        avatar_id (str): Unique identifier for the avatar.
        avatar_persona (AvatarPersona): Avatar persona configuration.
    """

    mode: str = "CUSTOM"
    avatar_id: str
    avatar_persona: Optional[AvatarPersona] = None
    livekit_config: Optional[CustomSDKLiveKitConfig] = None


class SessionTokenData(BaseModel):
    """Data model for session token response.

    Parameters:
        session_id (str): Unique identifier for the session.
        session_token (str): Session token for authentication.
    """

    session_id: str
    session_token: str


class SessionTokenResponse(BaseModel):
    """Response model for LiveAvatar session token.

    Parameters:
        code (int): Response status code.
        data (SessionTokenData): Session token data containing session_id and session_token.
        message (str): Response message.
    """

    code: int
    data: SessionTokenData
    message: str


class LiveAvatarSessionData(BaseModel):
    """Data model for LiveAvatar session response.

    Parameters:
        session_id (str): Unique identifier for the streaming session.
        livekit_url (str): LiveKit server URL for the session.
        livekit_client_token (str): Access token for LiveKit user.
        livekit_agent_token (str): Access token for LiveKit Agent (Pipecat).
        max_session_duration (int): Maximum session duration in seconds.
        ws_url (str): WebSocket URL for the session.
    """

    session_id: str
    livekit_url: str
    livekit_client_token: str
    livekit_agent_token: str
    max_session_duration: int
    ws_url: str


class LiveAvatarSessionResponse(BaseModel):
    """Response model for LiveAvatar session start.

    Parameters:
        code (int): Response status code.
        data (LiveAvatarSessionData): Session data containing connection details.
        message (str): Response message.
    """

    code: int
    data: LiveAvatarSessionData
    message: str


class LiveAvatarApiError(Exception):
    """Custom exception for LiveAvatar API errors."""

    def __init__(self, message: str, status: int, response_text: str) -> None:
        """Initialize the LiveAvatar API error.

        Args:
            message: Error message
            status: HTTP status code
            response_text: Raw response text from the API
        """
        super().__init__(message)
        self.status = status
        self.response_text = response_text


class LiveAvatarApi(BaseAvatarApi):
    """LiveAvatar Streaming API client."""

    BASE_URL = "https://api.liveavatar.com/v1"

    def __init__(self, api_key: str, session: aiohttp.ClientSession) -> None:
        """Initialize the LiveAvatar API.

        Args:
            api_key: LiveAvatar API key
            session: aiohttp client session
        """
        self._api_key = api_key
        self._session = session
        self._session_token = None

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        bearer_token: Optional[str] = None,
    ) -> Any:
        """Make a request to the LiveAvatar API.

        Args:
            method: HTTP method (GET, POST, etc.).
            path: API endpoint path.
            params: JSON-serializable parameters.
            bearer_token: Optional bearer token for authorization.

        Returns:
            Parsed JSON response data.

        Raises:
            LiveAvatarApiError: If the API response is not successful.
            aiohttp.ClientError: For network-related errors.
        """
        url = f"{self.BASE_URL}{path}"
        headers = {
            "accept": "application/json",
        }

        if bearer_token:
            headers["authorization"] = f"Bearer {bearer_token}"
        else:
            headers["X-API-KEY"] = self._api_key

        if params is not None:
            headers["content-type"] = "application/json"

        logger.debug(f"LiveAvatar API request: {method} {url}")

        try:
            async with self._session.request(method, url, json=params, headers=headers) as response:
                if not response.ok:
                    response_text = await response.text()
                    logger.error(f"LiveAvatar API error: {response_text}")
                    raise LiveAvatarApiError(
                        f"API request failed with status {response.status}",
                        response.status,
                        response_text,
                    )
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Network error while calling LiveAvatar API: {str(e)}")
            raise

    async def create_session_token(
        self, request_data: LiveAvatarNewSessionRequest
    ) -> SessionTokenResponse:
        """Create a session token for LiveAvatar.

        https://docs.liveavatar.com/reference/create_session_token_v1_sessions_token_post

        Args:
            request_data: Session token configuration parameters.

        Returns:
            Session token information.
        """
        params: dict[str, Any] = {
            "mode": request_data.mode,
            "avatar_id": request_data.avatar_id,
        }

        # Only include avatar_persona if it exists and has non-None values
        if request_data.avatar_persona is not None:
            avatar_persona = {
                "voice_id": request_data.avatar_persona.voice_id,
                "context_id": request_data.avatar_persona.context_id,
                "language": request_data.avatar_persona.language,
            }
            # Remove None values from avatar_persona
            avatar_persona = {k: v for k, v in avatar_persona.items() if v is not None}
            params["avatar_persona"] = avatar_persona

        response = await self._request("POST", "/sessions/token", params)
        logger.debug(f"LiveAvatar session token created")

        return SessionTokenResponse.model_validate(response)

    async def start_session(self, session_token: str) -> LiveAvatarSessionResponse:
        """Start a new LiveAvatar session.

        https://docs.liveavatar.com/reference/start_session_v1_sessions_start_post

        Args:
            session_token: Session token obtained from create_session_token.

        Returns:
            Session information including room URL and session ID.
        """
        response = await self._request("POST", "/sessions/start", bearer_token=session_token)
        logger.debug(f"LiveAvatar session started")

        return LiveAvatarSessionResponse.model_validate(response)

    async def stop_session(self, session_id: str, session_token: str) -> Any:
        """Stop an active LiveAvatar session.

        https://docs.liveavatar.com/reference/stop_session_v1_sessions_stop_post

        Args:
            session_id: ID of the session to stop.
            session_token: Session token for authentication.

        Returns:
            Response data from the stop session API call.

        Raises:
            ValueError: If session ID is not set.
        """
        if not session_id:
            raise ValueError("Session ID is not set.")

        params = {"session_id": session_id}

        response = await self._request(
            "POST", "/sessions/stop", params=params, bearer_token=session_token
        )
        return response

    async def new_session(
        self, request_data: LiveAvatarNewSessionRequest
    ) -> StandardSessionResponse:
        """Create and start a new LiveAvatar session (convenience method).

        This combines create_session_token and start_session into a single call.

        Args:
            request_data: Session token configuration parameters.

        Returns:
            StandardSessionResponse: Standardized session information with LiveAvatar raw response.
        """
        # Create session token
        token_response = await self.create_session_token(request_data)
        self._session_token = token_response.data.session_token

        # Start the session using the session_token from the data field
        session_response = await self.start_session(token_response.data.session_token)

        # Convert to standardized response
        return StandardSessionResponse(
            session_id=session_response.data.session_id,
            access_token=session_response.data.livekit_client_token,
            livekit_url=session_response.data.livekit_url,
            livekit_agent_token=session_response.data.livekit_agent_token,
            ws_url=session_response.data.ws_url,
            raw_response=session_response,
        )

    async def close_session(self, session_id: str) -> Any:
        """Close an active LiveAvatar session (convenience method).

        This is a convenience method that closes a session using the stored session token
        from the most recent `new_session()` call. It automatically uses the internally
        stored session token, eliminating the need to manually track tokens.

        Args:
            session_id: ID of the session to close.

        Returns:
            Response data from the stop session API call.

        Raises:
            ValueError: If no session token is available (i.e., `new_session()`
                       hasn't been called yet or the stored token is None).

        Note:
            This method requires that `new_session()` has been called previously to
            establish a stored session token. For more control over session tokens,
            use `stop_session()` directly with an explicit token parameter.
        """
        if not self._session_token:
            raise ValueError("Session token is not set. Call new_session first.")

        return await self.stop_session(session_id, self._session_token)
