#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Daily REST Helpers

Methods that wrap the Daily API to create rooms, check room URLs, and get meeting tokens.

"""
import urllib.parse
import urllib
import requests
from typing import Literal, Optional
from time import time

from pydantic import BaseModel, ValidationError


class DailyRoomSipParams(BaseModel):
    display_name: str = "sw-sip-dialin"
    video: bool = False
    sip_mode: str = "dial-in"
    num_endpoints: int = 1


class DailyRoomProperties(BaseModel):
    exp: float = time() + 5 * 60
    enable_chat: bool = False
    enable_emoji_reactions: bool = False
    eject_at_room_exp: bool = True
    enable_dialout: Optional[bool] = None
    sip: DailyRoomSipParams = None
    sip_uri: dict = None

    @property
    def sip_endpoint(self) -> str:
        return "sip:%s" % self.sip_uri['endpoint']


class DailyRoomParams(BaseModel):
    name: Optional[str] = None
    privacy: Literal['private', 'public'] = "public"
    properties: DailyRoomProperties = DailyRoomProperties()


class DailyRoomObject(BaseModel):
    id: str
    name: str
    api_created: bool
    privacy: str
    url: str
    created_at: str
    config: DailyRoomProperties


class DailyRESTHelper:
    def __init__(self, daily_api_key: str, daily_api_url: str = "api.daily.co/v1"):
        self.daily_api_key = daily_api_key
        self.daily_api_url = daily_api_url

    def _get_name_from_url(self, room_url: str) -> str:
        return urllib.parse.urlparse(room_url).path[1:]

    def create_room(self, params: DailyRoomParams) -> DailyRoomObject:
        res = requests.post(
            f"https://{self.daily_api_url}/rooms",
            headers={"Authorization": f"Bearer {self.daily_api_key}"},
            json={**params.model_dump(exclude_none=True)}
        )

        if res.status_code != 200:
            raise Exception(f"Unable to create room: {res.text}")

        data = res.json()

        try:
            room = DailyRoomObject(**data)
        except ValidationError as e:
            raise Exception(f"Invalid response: {e}")

        return room

    def _get_room_from_name(self, room_name: str) -> DailyRoomObject:
        res: requests.Response = requests.get(
            f"https://{self.daily_api_url}/rooms/{room_name}",
            headers={"Authorization": f"Bearer {self.daily_api_key}"}
        )

        if res.status_code != 200:
            raise Exception(f"Room not found: {room_name}")

        data = res.json()

        try:
            room = DailyRoomObject(**data)
        except ValidationError as e:
            raise Exception(f"Invalid response: {e}")

        return room

    def get_room_from_url(self, room_url: str,) -> DailyRoomObject:
        room_name = self._get_name_from_url(room_url)
        return self._get_room_from_name(room_name)

    def get_token(self, room_url: str, expiry_time: float = 60 * 60, owner: bool = True) -> str:
        if not room_url:
            raise Exception(
                "No Daily room specified. You must specify a Daily room in order a token to be generated.")

        expiration: float = time() + expiry_time

        room_name = self._get_name_from_url(room_url)

        res: requests.Response = requests.post(
            f"https://{self.daily_api_url}/meeting-tokens",
            headers={
                "Authorization": f"Bearer {self.daily_api_key}"},
            json={
                "properties": {
                    "room_name": room_name,
                    "is_owner": owner,
                    "exp": expiration
                }},
        )

        if res.status_code != 200:
            raise Exception(
                f"Failed to create meeting token: {res.status_code} {res.text}")

        token: str = res.json()["token"]

        return token
