#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pinch API.

API to communicate with Pinch Translation API.
"""

import os
from typing import Any, Dict, Literal, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel, field_validator


class PinchSessionRequest(BaseModel):
    """Request model for creating a new Pinch translation session.

    Parameters:
        source_language (str): Source language code (e.g., "en", "es", "fr")
        target_language (str): Target language code (e.g., "en", "es", "fr")
        voice_type (Literal["male", "female", "custom"]): Voice type for translation output
        voice_id (Optional[str]): Voice ID for custom voices (required when voice_type="custom")
        enable_audio_output (bool): Enable audio output from translation
        enable_video_output (bool): Enable video output (future feature)
    """

    source_language: str = "en"
    target_language: str = "es"
    voice_type: Literal["male", "female", "custom"] = "female"
    voice_id: Optional[str] = None
    enable_audio_output: bool = True
    enable_video_output: bool = False

    @field_validator("voice_id")
    @classmethod
    def validate_voice_id(cls, v, info):
        """Validate that voice_id is provided when voice_type is 'custom'."""
        if info.data.get("voice_type") == "custom" and not v:
            raise ValueError("voice_id is required when voice_type is 'custom'")
        return v

    @field_validator("source_language", "target_language")
    @classmethod
    def validate_language_codes(cls, v):
        """Validate language codes are not empty."""
        if not v or not v.strip():
            raise ValueError("Language code cannot be empty")
        # Basic validation - should be 2-3 letter codes
        if len(v.strip()) < 2 or len(v.strip()) > 3:
            raise ValueError("Language code should be 2-3 characters")
        return v.strip().lower()


class PinchSession(BaseModel):
    """Response model for a Pinch translation session.

    Parameters:
        session_id (str): Unique identifier for the translation session
        room_name (str): LiveKit room name
        livekit_url (str): LiveKit server URL
        access_token (str): LiveKit access token
        websocket_url (Optional[str]): WebSocket URL (if available)
    """

    session_id: str
    room_name: str
    livekit_url: str
    access_token: str
    websocket_url: Optional[str] = None


class PinchApiError(Exception):
    """Custom exception for Pinch API errors."""

    def __init__(self, message: str, status: int, response_text: str) -> None:
        """Initialize the Pinch API error.

        Args:
            message: Error message
            status: HTTP status code
            response_text: Raw response text from the API
        """
        super().__init__(message)
        self.status = status
        self.response_text = response_text


class PinchConnectionError(Exception):
    """Exception raised for LiveKit connection issues."""

    def __init__(self, message: str, cause: Exception = None) -> None:
        """Initialize the connection error.

        Args:
            message: Error message
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.cause = cause


class PinchConfigurationError(Exception):
    """Exception raised for configuration issues."""

    def __init__(self, message: str) -> None:
        """Initialize the configuration error.

        Args:
            message: Error message
        """
        super().__init__(message)


class PinchApi:
    """Pinch Translation API client.

    Note: In production you must set PINCH_API_BASE_URL environment variable.
    """

    DEFAULT_BASE_URL = "https://api.startpinch.com/api/beta1"

    def __init__(self, api_token: str, session: aiohttp.ClientSession) -> None:
        """Initialize the Pinch API.

        Args:
            api_token: Pinch API token
            session: aiohttp client session

        Raises:
            ValueError: If PINCH_API_BASE_URL is not set in production environments
        """
        self.api_token = api_token
        self.session = session
        self.base_url = os.getenv("PINCH_API_BASE_URL", self.DEFAULT_BASE_URL)

    async def _request(self, path: str, params: Dict[str, Any], expect_data: bool = True) -> Any:
        """Make a POST request to the Pinch API.

        Args:
            path: API endpoint path.
            params: JSON-serializable parameters.
            expect_data: Whether to expect JSON data in response (default: True).

        Returns:
            Parsed JSON response if expect_data is True, raw response text otherwise.

        Raises:
            PinchApiError: If the API response is not successful or data is missing when expected.
            aiohttp.ClientError: For network-related errors.
        """
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        try:
            async with self.session.post(url, json=params, headers=headers) as response:
                if not response.ok:
                    response_text = await response.text()
                    logger.error(f"Pinch API error: {response_text}")
                    raise PinchApiError(
                        f"API request failed with status {response.status}",
                        response.status,
                        response_text,
                    )
                if expect_data:
                    json_data = await response.json()
                    return json_data
                return await response.text()
        except aiohttp.ClientError as e:
            logger.error(f"Network error while calling Pinch API: {str(e)}")
            raise

    async def new_session(self, request_data: PinchSessionRequest) -> PinchSession:
        """Create a new translation session.

        Args:
            request_data: Session configuration parameters.

        Returns:
            Session information, including LiveKit connection details.
        """
        params = {
            "sourceLanguage": request_data.source_language,
            "targetLanguage": request_data.target_language,
            "voiceType": request_data.voice_type,
        }

        if request_data.voice_type == "custom" and request_data.voice_id:
            params["voiceId"] = request_data.voice_id

        if not request_data.enable_audio_output:
            params["enableAudioOutput"] = False

        if request_data.enable_video_output:
            params["enableVideoOutput"] = False

        session_info = await self._request("/session", params)

        if not session_info:
            raise PinchApiError("Empty response from Pinch API", 500, "No session data received")

        # Map response fields to our model - Pinch API returns: url, token, room_name
        session_data = {
            "session_id": session_info.get("room_name"),  # Use room_name as session_id
            "room_name": session_info.get("room_name"),
            "livekit_url": session_info.get("url"),
            "access_token": session_info.get("token"),
            "websocket_url": None,
        }
        # Validate required fields
        if not session_data.get("room_name"):
            raise PinchApiError("Missing room_name in API response", 500, str(session_info))
        if not session_data.get("livekit_url"):
            raise PinchApiError("Missing url in API response", 500, str(session_info))
        if not session_data.get("access_token"):
            raise PinchApiError("Missing token in API response", 500, str(session_info))

        return PinchSession.model_validate(session_data)

    async def end_session(self, session_id: str) -> None:
        """End a translation session.

        Args:
            session_id: The session ID to end.

        Raises:
            PinchApiError: If the API response is not successful.
            aiohttp.ClientError: For network-related errors.
        """
        params = {"session_id": session_id}
        await self._request("/session/end", params, expect_data=False)
