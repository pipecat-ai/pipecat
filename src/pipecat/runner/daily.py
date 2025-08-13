#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Daily room and token configuration utilities.

This module provides helper functions for creating and configuring Daily rooms
and authentication tokens. It automatically creates temporary rooms for
development or uses existing rooms specified via environment variables.

Environment variables:

- DAILY_API_KEY - Daily API key for room/token creation (required)
- DAILY_SAMPLE_ROOM_URL (optional) - Existing room URL to use. If not provided,
  a temporary room will be created automatically.

Example::

    import aiohttp
    from pipecat.runner.daily import configure

    async with aiohttp.ClientSession() as session:
        room_url, token = await configure(session)
        # Use room_url and token with DailyTransport
"""

import os
import time
import uuid
from typing import Tuple

import aiohttp

from pipecat.transports.services.helpers.daily_rest import (
    DailyRESTHelper,
    DailyRoomParams,
    DailyRoomProperties,
)


async def configure(aiohttp_session: aiohttp.ClientSession) -> Tuple[str, str]:
    """Configure Daily room URL and token from environment variables.

    This function will either:
    1. Use an existing room URL from DAILY_SAMPLE_ROOM_URL environment variable
    2. Create a new temporary room automatically if no URL is provided

    Args:
        aiohttp_session: HTTP session for making API requests.

    Returns:
        Tuple containing the room URL and authentication token.

    Raises:
        Exception: If DAILY_API_KEY is not provided in environment variables.
    """
    # Check for required API key
    api_key = os.getenv("DAILY_API_KEY")
    if not api_key:
        raise Exception(
            "DAILY_API_KEY environment variable is required. "
            "Get your API key from https://dashboard.daily.co/developers"
        )

    # Check for existing room URL
    existing_room_url = os.getenv("DAILY_SAMPLE_ROOM_URL")

    daily_rest_helper = DailyRESTHelper(
        daily_api_key=api_key,
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )

    if existing_room_url:
        # Use existing room
        print(f"Using existing Daily room: {existing_room_url}")
        room_url = existing_room_url
    else:
        # Create a new temporary room
        room_name = f"pipecat-{uuid.uuid4().hex[:8]}"
        print(f"Creating new Daily room: {room_name}")

        # Calculate expiration time: current time + 2 hours
        expiration_time = time.time() + (2 * 60 * 60)  # 2 hours from now

        # Create room properties with absolute timestamp
        room_properties = DailyRoomProperties(
            exp=expiration_time,  # Absolute Unix timestamp
            eject_at_room_exp=True,
        )

        # Create room parameters
        room_params = DailyRoomParams(name=room_name, properties=room_properties)

        room_response = await daily_rest_helper.create_room(room_params)
        room_url = room_response.url
        print(f"Created Daily room: {room_url}")

    # Create a meeting token for the room with an expiration 2 hours in the future
    expiry_time: float = 2 * 60 * 60
    token = await daily_rest_helper.get_token(room_url, expiry_time)

    return (room_url, token)


# Keep this for backwards compatibility, but mark as deprecated
async def configure_with_args(aiohttp_session: aiohttp.ClientSession, parser=None):
    """Configure Daily room with command-line argument parsing.

    .. deprecated:: 0.0.78
        This function is deprecated. Use configure() instead which uses
        environment variables only.

    Args:
        aiohttp_session: HTTP session for making API requests.
        parser: Ignored. Kept for backwards compatibility.

    Returns:
        Tuple containing room URL, authentication token, and None (for args).
    """
    import warnings

    warnings.warn(
        "configure_with_args is deprecated. Use configure() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    room_url, token = await configure(aiohttp_session)
    return (room_url, token, None)
