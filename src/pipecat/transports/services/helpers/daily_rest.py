#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Daily REST Helpers.

Methods that wrap the Daily API to create rooms, check room URLs, and get meeting tokens.
"""

import time
from typing import Literal, Optional
from urllib.parse import urlparse

import aiohttp
from pydantic import BaseModel, Field, ValidationError


class DailyRoomSipParams(BaseModel):
    """SIP configuration parameters for Daily rooms.

    Parameters:
        display_name: Name shown for the SIP endpoint.
        video: Whether video is enabled for SIP.
        sip_mode: SIP connection mode, typically 'dial-in'.
        num_endpoints: Number of allowed SIP endpoints.
    """

    display_name: str = "sw-sip-dialin"
    video: bool = False
    sip_mode: str = "dial-in"
    num_endpoints: int = 1


class RecordingsBucketConfig(BaseModel):
    """Configuration for storing Daily recordings in a custom S3 bucket.

    Refer to the Daily API documentation for more information:
    https://docs.daily.co/guides/products/live-streaming-recording/storing-recordings-in-a-custom-s3-bucket

    Parameters:
        bucket_name: Name of the S3 bucket for storing recordings.
        bucket_region: AWS region where the S3 bucket is located.
        assume_role_arn: ARN of the IAM role to assume for S3 access.
        allow_api_access: Whether to allow API access to the recordings.
    """

    bucket_name: str
    bucket_region: str
    assume_role_arn: str
    allow_api_access: bool = False


class TranscriptionBucketConfig(BaseModel):
    """Configuration for storing Daily transcription in a custom S3 bucket.

    Refer to the Daily API documentation for more information:
    https://docs.daily.co/guides/products/live-streaming-recording/storing-recordings-in-a-custom-s3-bucket

    Parameters:
        bucket_name: Name of the S3 bucket for storing transcription.
        bucket_region: AWS region where the S3 bucket is located.
        assume_role_arn: ARN of the IAM role to assume for S3 access.
        allow_api_access: Whether to allow API access to the transcription.
    """

    bucket_name: str
    bucket_region: str
    assume_role_arn: str
    allow_api_access: bool = False


class DailyRoomProperties(BaseModel, extra="allow"):
    """Properties for configuring a Daily room.

    Reference: https://docs.daily.co/reference/rest-api/rooms/create-room#properties

    Parameters:
        exp: Optional Unix epoch timestamp for room expiration (e.g., time.time() + 300 for 5 minutes).
        enable_chat: Whether chat is enabled in the room.
        enable_prejoin_ui: Whether the pre-join UI is enabled.
        enable_emoji_reactions: Whether emoji reactions are enabled.
        eject_at_room_exp: Whether to remove participants when room expires.
        enable_dialout: Whether SIP dial-out is enabled.
        enable_recording: Recording settings ('cloud', 'local', 'raw-tracks').
        enable_transcription_storage: Whether transcription storage is enabled.
        geo: Geographic region for room.
        max_participants: Maximum number of participants allowed in the room.
        recordings_bucket: Configuration for custom S3 bucket recordings.
        transcription_bucket: Configuration for custom S3 bucket transcription.
        sip: SIP configuration parameters.
        sip_uri: SIP URI information returned by Daily.
        start_video_off: Whether video is off by default.
    """

    exp: Optional[float] = None
    enable_chat: bool = False
    enable_prejoin_ui: bool = False
    enable_emoji_reactions: bool = False
    eject_at_room_exp: bool = False
    enable_dialout: Optional[bool] = None
    enable_recording: Optional[Literal["cloud", "local", "raw-tracks"]] = None
    enable_transcription_storage: Optional[bool] = None
    geo: Optional[str] = None
    max_participants: Optional[int] = None
    recordings_bucket: Optional[RecordingsBucketConfig] = None
    transcription_bucket: Optional[TranscriptionBucketConfig] = None
    sip: Optional[DailyRoomSipParams] = None
    sip_uri: Optional[dict] = None
    start_video_off: bool = False

    @property
    def sip_endpoint(self) -> str:
        """Get the SIP endpoint URI if available.

        Returns:
            SIP endpoint URI or empty string if not available.
        """
        if not self.sip_uri:
            return ""
        else:
            return "sip:%s" % self.sip_uri["endpoint"]


class DailyRoomParams(BaseModel):
    """Parameters for creating a Daily room.

    Parameters:
        name: Optional custom name for the room.
        privacy: Room privacy setting ('private' or 'public').
        properties: Room configuration properties.
    """

    name: Optional[str] = None
    privacy: Literal["private", "public"] = "public"
    properties: DailyRoomProperties = Field(default_factory=DailyRoomProperties)


class DailyRoomObject(BaseModel):
    """Represents a Daily room returned by the API.

    Parameters:
        id: Unique room identifier.
        name: Room name.
        api_created: Whether room was created via API.
        privacy: Room privacy setting ('private' or 'public').
        url: Full URL for joining the room.
        created_at: Timestamp of room creation in ISO 8601 format (e.g., "2019-01-26T09:01:22.000Z").
        config: Room configuration properties.
    """

    id: str
    name: str
    api_created: bool
    privacy: str
    url: str
    created_at: str
    config: DailyRoomProperties


class DailyMeetingTokenProperties(BaseModel):
    """Properties for configuring a Daily meeting token.

    Refer to the Daily API documentation for more information:
    https://docs.daily.co/reference/rest-api/meeting-tokens/create-meeting-token#properties

    Parameters:
        room_name: The room for which this token is valid. If not set, the token is valid for all rooms in your domain.
        eject_at_token_exp: If True, the user will be ejected from the room when the token expires.
        eject_after_elapsed: The number of seconds after which the user will be ejected from the room.
        nbf: Not before timestamp - users cannot join with this token before this time.
        exp: Expiration time (unix timestamp in seconds). Strongly recommended for security.
        is_owner: If True, the token will grant owner privileges in the room.
        user_name: The name of the user. This will be added to the token payload.
        user_id: A unique identifier for the user. This will be added to the token payload.
        enable_screenshare: If True, the user will be able to share their screen.
        start_video_off: If True, the user's video will be turned off when they join the room.
        start_audio_off: If True, the user's audio will be turned off when they join the room.
        enable_recording: Recording settings for the token. Must be one of 'cloud', 'local' or 'raw-tracks'.
        enable_prejoin_ui: If True, the user will see the prejoin UI before joining the room.
        start_cloud_recording: Start cloud recording when the user joins the room.
        permissions: Specifies the initial default permissions for a non-meeting-owner participant.
    """

    room_name: Optional[str] = None
    eject_at_token_exp: Optional[bool] = None
    eject_after_elapsed: Optional[int] = None
    nbf: Optional[int] = None
    exp: Optional[int] = None
    is_owner: Optional[bool] = None
    user_name: Optional[str] = None
    user_id: Optional[str] = None
    enable_screenshare: Optional[bool] = None
    start_video_off: Optional[bool] = None
    start_audio_off: Optional[bool] = None
    enable_recording: Optional[Literal["cloud", "local", "raw-tracks"]] = None
    enable_prejoin_ui: Optional[bool] = None
    start_cloud_recording: Optional[bool] = None
    permissions: Optional[dict] = None


class DailyMeetingTokenParams(BaseModel):
    """Parameters for creating a Daily meeting token.

    Refer to the Daily API documentation for more information:
    https://docs.daily.co/reference/rest-api/meeting-tokens/create-meeting-token#body-params

    Parameters:
        properties: Meeting token configuration properties.
    """

    properties: DailyMeetingTokenProperties = Field(default_factory=DailyMeetingTokenProperties)


class DailyRESTHelper:
    """Helper class for interacting with Daily's REST API.

    Provides methods for creating, managing, and accessing Daily rooms.
    """

    def __init__(
        self,
        *,
        daily_api_key: str,
        daily_api_url: str = "https://api.daily.co/v1",
        aiohttp_session: aiohttp.ClientSession,
    ):
        """Initialize the Daily REST helper.

        Args:
            daily_api_key: Your Daily API key.
            daily_api_url: Daily API base URL (e.g. "https://api.daily.co/v1").
            aiohttp_session: Async HTTP session for making requests.
        """
        self.daily_api_key = daily_api_key
        self.daily_api_url = daily_api_url
        self.aiohttp_session = aiohttp_session

    def get_name_from_url(self, room_url: str) -> str:
        """Extract room name from a Daily room URL.

        Args:
            room_url: Full Daily room URL.

        Returns:
            Room name portion of the URL.
        """
        return urlparse(room_url).path[1:]

    async def get_room_from_url(self, room_url: str) -> DailyRoomObject:
        """Get room details from a Daily room URL.

        Args:
            room_url: Full Daily room URL.

        Returns:
            DailyRoomObject instance for the room.
        """
        room_name = self.get_name_from_url(room_url)
        return await self._get_room_from_name(room_name)

    async def create_room(self, params: DailyRoomParams) -> DailyRoomObject:
        """Create a new Daily room.

        Args:
            params: Room configuration parameters.

        Returns:
            DailyRoomObject instance for the created room.

        Raises:
            Exception: If room creation fails or response is invalid.
        """
        headers = {"Authorization": f"Bearer {self.daily_api_key}"}
        json = params.model_dump(exclude_none=True)
        async with self.aiohttp_session.post(
            f"{self.daily_api_url}/rooms", headers=headers, json=json
        ) as r:
            if r.status != 200:
                text = await r.text()
                raise Exception(f"Unable to create room (status: {r.status}): {text}")

            data = await r.json()

        try:
            room = DailyRoomObject(**data)
        except ValidationError as e:
            raise Exception(f"Invalid response: {e}")

        return room

    async def get_token(
        self,
        room_url: str,
        expiry_time: float = 60 * 60,
        eject_at_token_exp: bool = False,
        owner: bool = True,
        params: Optional[DailyMeetingTokenParams] = None,
    ) -> str:
        """Generate a meeting token for user to join a Daily room.

        Args:
            room_url: Daily room URL.
            expiry_time: Token validity duration in seconds (default: 1 hour).
            eject_at_token_exp: Whether to eject user when token expires.
            owner: Whether token has owner privileges.
            params: Optional additional token properties. Note that room_name,
                exp, and is_owner will be set based on the other function
                parameters regardless of values in params.

        Returns:
            Meeting token.

        Raises:
            Exception: If token generation fails or room URL is missing.
        """
        if not room_url:
            raise Exception(
                "No Daily room specified. You must specify a Daily room in order a token to be generated."
            )

        expiration: int = int(time.time() + expiry_time)

        room_name = self.get_name_from_url(room_url)

        headers = {"Authorization": f"Bearer {self.daily_api_key}"}

        if params is None:
            params = DailyMeetingTokenParams(
                properties=DailyMeetingTokenProperties(
                    room_name=room_name,
                    is_owner=owner,
                    exp=expiration,
                    eject_at_token_exp=eject_at_token_exp,
                )
            )
        else:
            params.properties.room_name = room_name
            params.properties.exp = expiration
            params.properties.eject_at_token_exp = eject_at_token_exp
            params.properties.is_owner = owner

        json = params.model_dump(exclude_none=True)

        async with self.aiohttp_session.post(
            f"{self.daily_api_url}/meeting-tokens", headers=headers, json=json
        ) as r:
            if r.status != 200:
                text = await r.text()
                raise Exception(f"Failed to create meeting token (status: {r.status}): {text}")

            data = await r.json()

        return data["token"]

    async def delete_room_by_url(self, room_url: str) -> bool:
        """Delete a room using its URL.

        Args:
            room_url: Daily room URL.

        Returns:
            True if deletion was successful.
        """
        room_name = self.get_name_from_url(room_url)
        return await self.delete_room_by_name(room_name)

    async def delete_room_by_name(self, room_name: str) -> bool:
        """Delete a room using its name.

        Args:
            room_name: Name of the room to delete.

        Returns:
            True if deletion was successful.

        Raises:
            Exception: If deletion fails (excluding 404 Not Found).
        """
        headers = {"Authorization": f"Bearer {self.daily_api_key}"}
        async with self.aiohttp_session.delete(
            f"{self.daily_api_url}/rooms/{room_name}", headers=headers
        ) as r:
            if r.status != 200 and r.status != 404:
                text = await r.text()
                raise Exception(f"Failed to delete room [{room_name}] (status: {r.status}): {text}")

        return True

    async def _get_room_from_name(self, room_name: str) -> DailyRoomObject:
        """Internal method to get room details by name."""
        headers = {"Authorization": f"Bearer {self.daily_api_key}"}
        async with self.aiohttp_session.get(
            f"{self.daily_api_url}/rooms/{room_name}", headers=headers
        ) as r:
            if r.status != 200:
                raise Exception(f"Room not found: {room_name}")

            data = await r.json()

        try:
            room = DailyRoomObject(**data)
        except ValidationError as e:
            raise Exception(f"Invalid response: {e}")

        return room
