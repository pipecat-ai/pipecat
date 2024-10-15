#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Daily REST Helpers

Methods that wrap the Daily API to create rooms, check room URLs, and get meeting tokens.

"""

import aiohttp
import time

from urllib.parse import urlparse

from pydantic import Field, BaseModel, ValidationError
from typing import Literal, Optional


class DailyRoomSipParams(BaseModel):
    display_name: str = "sw-sip-dialin"
    video: bool = False
    sip_mode: str = "dial-in"
    num_endpoints: int = 1


class DailyRoomProperties(BaseModel, extra="allow"):
    exp: float = Field(default_factory=lambda: time.time() + 5 * 60)
    enable_chat: bool = False
    enable_emoji_reactions: bool = False
    eject_at_room_exp: bool = True
    enable_dialout: Optional[bool] = None
    sip: Optional[DailyRoomSipParams] = None
    sip_uri: Optional[dict] = None

    @property
    def sip_endpoint(self) -> str:
        if not self.sip_uri:
            return ""
        else:
            return "sip:%s" % self.sip_uri["endpoint"]


class DailyRoomParams(BaseModel):
    name: Optional[str] = None
    privacy: Literal["private", "public"] = "public"
    properties: DailyRoomProperties = Field(default_factory=DailyRoomProperties)


class DailyRoomObject(BaseModel):
    id: str
    name: str
    api_created: bool
    privacy: str
    url: str
    created_at: str
    config: DailyRoomProperties


class DailyRESTHelper:
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
        return urlparse(room_url).path[1:]

    async def get_room_from_url(self, room_url: str) -> DailyRoomObject:
        room_name = self.get_name_from_url(room_url)
        return await self._get_room_from_name(room_name)

    async def create_room(self, params: DailyRoomParams) -> DailyRoomObject:
        headers = {"Authorization": f"Bearer {self.daily_api_key}"}
        json = {**params.model_dump(exclude_none=True)}
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

    async def delete_room_by_url(self, room_url: str) -> bool:
        room_name = self.get_name_from_url(room_url)
        return await self.delete_room_by_name(room_name)

    async def delete_room_by_name(self, room_name: str) -> bool:
        headers = {"Authorization": f"Bearer {self.daily_api_key}"}
        async with self.aiohttp_session.delete(
            f"{self.daily_api_url}/rooms/{room_name}", headers=headers
        ) as r:
            if r.status != 200 and r.status != 404:
                text = await r.text()
                raise Exception(f"Failed to delete room [{room_name}] (status: {r.status}): {text}")

        return True

    async def _get_room_from_name(self, room_name: str) -> DailyRoomObject:
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
