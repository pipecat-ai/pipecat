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
from pydantic import BaseModel, Field, ValidationError, constr


class DailyRoomSipParams(BaseModel):
    """SIP configuration parameters for Daily rooms.

    Attributes:
        display_name: Name shown for the SIP endpoint
        video: Whether video is enabled for SIP
        sip_mode: SIP connection mode, typically 'dial-in'
        num_endpoints: Number of allowed SIP endpoints
    """

    display_name: str = "sw-sip-dialin"
    video: bool = False
    sip_mode: str = "dial-in"
    num_endpoints: int = 1


class DailyRoomProperties(BaseModel, extra="allow"):
    """Properties for configuring a Daily room.

    Attributes:
        exp: Optional Unix epoch timestamp for room expiration (e.g., time.time() + 300 for 5 minutes)
        enable_chat: Whether chat is enabled in the room
        enable_prejoin_ui: Whether the pre-join UI is enabled
        enable_emoji_reactions: Whether emoji reactions are enabled
        eject_at_room_exp: Whether to remove participants when room expires
        enable_dialout: Whether SIP dial-out is enabled
        max_participants: Maximum number of participants allowed in the room
        sip: SIP configuration parameters
        sip_uri: SIP URI information returned by Daily
        start_video_off: Whether video is off by default

    Reference: https://docs.daily.co/reference/rest-api/rooms/create-room#properties
    """

    exp: Optional[float] = None
    enable_chat: bool = False
    enable_prejoin_ui: bool = True
    enable_emoji_reactions: bool = False
    eject_at_room_exp: bool = True
    enable_dialout: Optional[bool] = None
    max_participants: Optional[int] = None
    sip: Optional[DailyRoomSipParams] = None
    sip_uri: Optional[dict] = None
    start_video_off: bool = False

    @property
    def sip_endpoint(self) -> str:
        """Get the SIP endpoint URI if available.

        Returns:
            str: SIP endpoint URI or empty string if not available
        """
        if not self.sip_uri:
            return ""
        else:
            return "sip:%s" % self.sip_uri["endpoint"]


class DailyRoomParams(BaseModel):
    """Parameters for creating a Daily room.

    Attributes:
        name: Optional custom name for the room
        privacy: Room privacy setting ('private' or 'public')
        properties: Room configuration properties
    """

    name: Optional[str] = None
    privacy: Literal["private", "public"] = "public"
    properties: DailyRoomProperties = Field(default_factory=DailyRoomProperties)


class DailyRoomObject(BaseModel):
    """Represents a Daily room returned by the API.

    Attributes:
        id: Unique room identifier
        name: Room name
        api_created: Whether room was created via API
        privacy: Room privacy setting ('private' or 'public')
        url: Full URL for joining the room
        created_at: Timestamp of room creation in ISO 8601 format (e.g., "2019-01-26T09:01:22.000Z").
        config: Room configuration properties
    """

    id: str
    name: str
    api_created: bool
    privacy: str
    url: str
    created_at: str
    config: DailyRoomProperties


class DailyMeetingTokenProperties(BaseModel):
    # https://docs.daily.co/reference/rest-api/meeting-tokens/create-meeting-token#properties

    room_name: Optional[str] = Field(
        default=None,
        description="The name of the room this token is for. If not provided, the token can be used for any room.",
    )

    eject_at_token_exp: Optional[bool] = Field(
        default=None,
        description="If `true`, the user will be ejected from the room when the token expires. Defaults to `false`.",
    )
    eject_after_elapsed: Optional[int] = Field(
        default=None,
        description="The number of seconds after which the user will be ejected from the room. If not provided, the user will not be ejected based on elapsed time.",
    )

    nfb: Optional[int] = Field(
        default=None,
        description="Not before. This is a unix timestamp (seconds since the epoch.) Users cannot join a meeting in with this token before this time.",
    )

    exp: Optional[int] = Field(
        default=60 * 60,
        description="Expiration time for the token in seconds since the epoch. If not provided, the token will not expire.",
    )
    is_owner: Optional[bool] = Field(
        default=True,
        description="If `true`, the token will grant owner privileges in the room. Defaults to `false`.",
    )
    user_name: Optional[str] = Field(
        default=None,
        description="The name of the user. This will be added to the token payload.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="A unique identifier for the user. This will be added to the token payload.",
    )
    enable_screenshare: Optional[bool] = Field(
        default=True,
        description="If `true`, the user will be able to share their screen. Defaults to `true`.",
    )
    start_video_off: Optional[bool] = Field(
        default=None,
        description="If `true`, the user's video will be turned off when they join the room. Defaults to `false`.",
    )
    start_audio_off: Optional[bool] = Field(
        default=None,
        description="If `true`, the user's audio will be turned off when they join the room. Defaults to `false`.",
    )
    enable_recording: Optional[Literal["cloud", "local", "raw-tracks"]] = Field(
        default=None,
        description="Recording settings for the token. Must be one of `cloud`, `local`, `raw-tracks` or `none`.",
    )
    enable_prejoin_ui: Optional[bool] = Field(
        default=None,
        description="If `true`, the user will see the prejoin UI before joining the room.",
    )
    start_cloud_recording: Optional[bool] = Field(
        default=None,
        description="Start cloud recording when the user joins the room. This can be used to always record and archive meetings, for example in a customer support context.",
    )


class DailyMeetingTokenParams(BaseModel):
    # https://docs.daily.co/reference/rest-api/meeting-tokens/create-meeting-token#body-params
    properties: DailyMeetingTokenProperties = Field(default_factory=DailyMeetingTokenProperties)


class DailyRESTHelper:
    """Helper class for interacting with Daily's REST API.

    Provides methods for creating, managing, and accessing Daily rooms.

    Args:
        daily_api_key: Your Daily API key
        daily_api_url: Daily API base URL (e.g. "https://api.daily.co/v1")
        aiohttp_session: Async HTTP session for making requests
    """

    def __init__(
        self,
        *,
        daily_api_key: str,
        daily_api_url: str = "https://api.daily.co/v1",
        aiohttp_session: aiohttp.ClientSession,
    ):
        self.daily_api_key = daily_api_key
        self.daily_api_url = daily_api_url
        self.aiohttp_session = aiohttp_session

    def get_name_from_url(self, room_url: str) -> str:
        """Extract room name from a Daily room URL.

        Args:
            room_url: Full Daily room URL

        Returns:
            str: Room name portion of the URL
        """
        return urlparse(room_url).path[1:]

    async def get_room_from_url(self, room_url: str) -> DailyRoomObject:
        """Get room details from a Daily room URL.

        Args:
            room_url: Full Daily room URL

        Returns:
            DailyRoomObject: DailyRoomObject instance for the room
        """
        room_name = self.get_name_from_url(room_url)
        return await self._get_room_from_name(room_name)

    async def create_room(self, params: DailyRoomParams) -> DailyRoomObject:
        """Create a new Daily room.

        Args:
            params: Room configuration parameters

        Returns:
            DailyRoomObject: DailyRoomObject instance for the created room

        Raises:
            Exception: If room creation fails or response is invalid
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
        self, room_url: str, expiry_time: float = 60 * 60, owner: bool = True
    ) -> str:
        """Generate a meeting token for user to join a Daily room.

        Args:
            room_url: Daily room URL
            expiry_time: Token validity duration in seconds (default: 1 hour)
            owner: Whether token has owner privileges

        Returns:
            str: Meeting token

        Raises:
            Exception: If token generation fails or room URL is missing
        """
        if not room_url:
            raise Exception(
                "No Daily room specified. You must specify a Daily room in order a token to be generated."
            )

        expiration: float = time.time() + expiry_time

        room_name = self.get_name_from_url(room_url)

        headers = {"Authorization": f"Bearer {self.daily_api_key}"}
        json = {"properties": {"room_name": room_name, "is_owner": owner, "exp": expiration}}
        async with self.aiohttp_session.post(
            f"{self.daily_api_url}/meeting-tokens", headers=headers, json=json
        ) as r:
            if r.status != 200:
                text = await r.text()
                raise Exception(f"Failed to create meeting token (status: {r.status}): {text}")

            data = await r.json()

        return data["token"]

    async def get_token_v2(
        self,
        room_url: str,
        params: DailyMeetingTokenParams,
        expiry_time: float = 60 * 60,
    ) -> str:
        """Generate a meeting token for user to join a Daily room.

        Args:
            room_url: Daily room URL
            params: Meeting token properties
            expiry_time: Token expiry time in seconds (default: 1 hour)

        Returns:
            str: Meeting token

        Raises:
            Exception: If token generation fails or room URL is missing
        """
        if not room_url:
            raise Exception(
                "No Daily room specified. You must specify a Daily room in order a token to be generated."
            )

        expiration: int = int(time.time() + expiry_time)

        room_name = self.get_name_from_url(room_url)

        params.properties.room_name = room_name
        params.properties.exp = expiration

        headers = {"Authorization": f"Bearer {self.daily_api_key}"}
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
            room_url: Daily room URL

        Returns:
            bool: True if deletion was successful
        """
        room_name = self.get_name_from_url(room_url)
        return await self.delete_room_by_name(room_name)

    async def delete_room_by_name(self, room_name: str) -> bool:
        """Delete a room using its name.

        Args:
            room_name: Name of the room to delete

        Returns:
            bool: True if deletion was successful

        Raises:
            Exception: If deletion fails (excluding 404 Not Found)
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
        """Internal method to get room details by name.

        Args:
            room_name: Name of the room

        Returns:
            DailyRoomObject: DailyRoomObject instance for the room

        Raises:
            Exception: If room is not found or response is invalid
        """
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
