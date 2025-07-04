from enum import Enum
from typing import Any, Dict, Literal, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field


class AvatarQuality(str, Enum):
    """Enum representing different avatar quality levels."""

    low = "low"
    medium = "medium"
    high = "high"


class VoiceEmotion(str, Enum):
    """Enum representing different voice emotion types."""

    EXCITED = "excited"
    SERIOUS = "serious"
    FRIENDLY = "friendly"
    SOOTHING = "soothing"
    BROADCASTER = "broadcaster"


class ElevenLabsSettings(BaseModel):
    """Settings for ElevenLabs voice configuration."""

    stability: Optional[float] = None
    similarity_boost: Optional[float] = None
    style: Optional[int] = None
    use_speaker_boost: Optional[bool] = None


class VoiceSettings(BaseModel):
    """Voice configuration settings."""

    voice_id: Optional[str] = Field(None, alias="voiceId")
    rate: Optional[float] = None
    emotion: Optional[VoiceEmotion] = None
    elevenlabs_settings: Optional[ElevenLabsSettings] = Field(None, alias="elevenlabsSettings")


class NewSessionRequest(BaseModel):
    """Request model for creating a new HeyGen session."""

    avatarName: str
    quality: Optional[AvatarQuality] = None
    knowledgeId: Optional[str] = None
    knowledgeBase: Optional[str] = None
    voice: Optional[VoiceSettings] = None
    language: Optional[str] = None
    version: Literal["v2"] = "v2"
    video_encoding: Literal["H264"] = "H264"
    source: Literal["sdk"] = "sdk"
    disableIdleTimeout: Optional[bool] = None


class HeyGenSession(BaseModel):
    """Response model for a HeyGen session."""

    session_id: str
    access_token: str
    realtime_endpoint: str
    url: str


class HeygenApiError(Exception):
    """Custom exception for HeyGen API errors."""

    def __init__(self, message: str, status: int, response_text: str) -> None:
        """Args:
        message: Error message
        status: HTTP status code
        response_text: Raw response text from the API
        """
        super().__init__(message)
        self.status = status
        self.response_text = response_text


class HeyGenApi:
    """HeyGen Streaming API client."""

    BASE_URL = "https://api.heygen.com/v1"

    def __init__(self, api_key: str, session: aiohttp.ClientSession) -> None:
        """Args:
        api_key: HeyGen API key
        session: Optional aiohttp client session
        """
        self.api_key = api_key
        self.session = session

    async def _request(self, path: str, params: Dict[str, Any], expect_data: bool = True) -> Any:
        """Make a POST request to the HeyGen API.

        Args:
            path: API endpoint path
            params: JSON-serializable parameters
            expect_data: Whether to expect and extract 'data' field from response (default: True)

        Returns:
            Parsed JSON response data

        Raises:
            HeygenApiError: If the API response is not successful or data is missing when expected
            aiohttp.ClientError: For network-related errors
        """
        url = f"{self.BASE_URL}{path}"
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        logger.debug(f"HeyGen API request: {url}")

        try:
            async with self.session.post(url, json=params, headers=headers) as response:
                if not response.ok:
                    response_text = await response.text()
                    logger.error(f"HeyGen API error: {response_text}")
                    raise HeygenApiError(
                        f"API request failed with status {response.status}",
                        response.status,
                        response_text,
                    )
                if expect_data:
                    json_data = await response.json()
                    data = json_data.get("data")
                    return data
                return await response.text()
        except aiohttp.ClientError as e:
            logger.error(f"Network error while calling HeyGen API: {str(e)}")
            raise

    async def new_session(self, request_data: NewSessionRequest) -> HeyGenSession:
        """Create a new streaming session.
        https://docs.heygen.com/reference/new-session

        Args:
            request_data: Session configuration parameters

        Returns:
            Session information including ID and access token
        """
        params = {
            "avatar_name": request_data.avatarName,
            "quality": request_data.quality,
            "knowledge_base_id": request_data.knowledgeId,
            "knowledge_base": request_data.knowledgeBase,
            "voice": {
                "voice_id": request_data.voice.voiceId if request_data.voice else None,
                "rate": request_data.voice.rate if request_data.voice else None,
                "emotion": request_data.voice.emotion if request_data.voice else None,
                "elevenlabs_settings": (
                    request_data.voice.elevenlabsSettings if request_data.voice else None
                ),
            },
            "language": request_data.language,
            "version": "v2",
            "video_encoding": "H264",
            "source": "sdk",
            "disable_idle_timeout": request_data.disableIdleTimeout,
        }
        session_info = await self._request("/streaming.new", params)
        print("heygen session info", session_info)

        return HeyGenSession.model_validate(session_info)

    async def start_session(self, session_id: str) -> Any:
        """Start the streaming session.
        https://docs.heygen.com/reference/start-session

        Args:
            session_id: ID of the session to start

        Returns:
            Response data from the start session API call

        Raises:
            ValueError: If session ID is not set
        """
        if not session_id:
            raise ValueError("Session ID is not set. Call new_session first.")

        params = {
            "session_id": session_id,
        }
        return await self._request("/streaming.start", params)

    async def close_session(self, session_id: str) -> Any:
        """Terminate an active the streaming session.
        https://docs.heygen.com/reference/close-session

        Args:
            session_id: ID of the session to stop

        Returns:
            Response data from the stop session API call

        Raises:
            ValueError: If session ID is not set
        """
        if not session_id:
            raise ValueError("Session ID is not set. Call new_session first.")

        params = {
            "session_id": session_id,
        }
        return await self._request("/streaming.stop", params, expect_data=False)
