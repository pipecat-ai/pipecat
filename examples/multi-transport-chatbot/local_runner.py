#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

import aiohttp

from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomParams


async def configure(aiohttp_session: aiohttp.ClientSession):
    (url, token) = await configure_with_args(aiohttp_session)
    return (url, token)


async def configure_with_args(aiohttp_session: aiohttp.ClientSession = None):
    key = os.getenv("DAILY_API_KEY")
    if not key:
        raise Exception(
            "No Daily API key specified. set DAILY_API_KEY in your environment to specify a Daily API key, available from https://dashboard.daily.co/developers."
        )

    daily_rest_helper = DailyRESTHelper(
        daily_api_key=key,
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )

    room = await daily_rest_helper.create_room(
        DailyRoomParams(properties={"enable_prejoin_ui": False})
    )
    if not room.url:
        raise HTTPException(status_code=500, detail="Failed to create room")

    url = room.url

    # Create a meeting token for the given room with an expiration 1 hour in
    # the future.
    expiry_time: float = 60 * 60

    token = await daily_rest_helper.get_token(url, expiry_time)

    return (url, token)
