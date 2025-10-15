#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Daily room and token configuration utilities.

This module provides helper functions for creating and configuring Daily rooms
and authentication tokens. It automatically creates temporary rooms for
development or uses existing rooms specified via environment variables.

Functions:

- configure(): Create a standard or SIP-enabled Daily room, returning a DailyRoomConfig object.

Environment variables:

- DAILY_API_KEY - Daily API key for room/token creation (required)
- DAILY_SAMPLE_ROOM_URL (optional) - Existing room URL to use. If not provided,
  a temporary room will be created automatically.

Example::

    import aiohttp
    from pipecat.runner.daily import configure

    async with aiohttp.ClientSession() as session:
        # Standard room
        room_url, token = await configure(session)

        # SIP-enabled room for phone calls
        config = await configure(session, sip_caller_phone="+15551234567")
        # config contains: room_url, token, sip_endpoint
"""

import os
import time
import uuid
from typing import Dict, List, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.transports.daily.utils import (
    DailyRESTHelper,
    DailyRoomParams,
    DailyRoomProperties,
    DailyRoomSipParams,
)


class DailyRoomConfig(BaseModel):
    """Configuration returned when creating a Daily room.

    Parameters:
        room_url: The Daily room URL for joining the meeting.
        token: Authentication token for the bot to join the room.
        sip_endpoint: SIP endpoint URI for phone connections (None for standard rooms).
    """

    room_url: str
    token: str
    sip_endpoint: Optional[str] = None

    def __iter__(self):
        """Enable tuple unpacking for backward compatibility.

        Allows: room_url, token = await configure(session)
        """
        yield self.room_url
        yield self.token


async def configure(
    aiohttp_session: aiohttp.ClientSession,
    *,
    room_exp_duration: Optional[float] = 2.0,
    token_exp_duration: Optional[float] = 2.0,
    sip_caller_phone: Optional[str] = None,
    sip_enable_video: Optional[bool] = False,
    sip_num_endpoints: Optional[int] = 1,
    sip_codecs: Optional[Dict[str, List[str]]] = None,
    room_properties: Optional[DailyRoomProperties] = None,
) -> DailyRoomConfig:
    """Configure Daily room URL and token with optional SIP capabilities.

    This function will either:
    1. Use an existing room URL from DAILY_SAMPLE_ROOM_URL environment variable (standard mode only)
    2. Create a new temporary room automatically if no URL is provided

    Args:
        aiohttp_session: HTTP session for making API requests.
        room_exp_duration: Room expiration time in hours.
        token_exp_duration: Token expiration time in hours.
        sip_caller_phone: Phone number or identifier for SIP display name.
            When provided, enables SIP functionality and returns SipRoomConfig.
        sip_enable_video: Whether video is enabled for SIP.
        sip_num_endpoints: Number of allowed SIP endpoints.
        sip_codecs: Codecs to support for audio and video. If None, uses Daily defaults.
            Example: {"audio": ["OPUS"], "video": ["H264"]}
        room_properties: Optional DailyRoomProperties to use instead of building from
            individual parameters. When provided, this overrides room_exp_duration and
            SIP-related parameters. If not provided, properties are built from the
            individual parameters as before.

    Returns:
        DailyRoomConfig: Object with room_url, token, and optional sip_endpoint.
        Supports tuple unpacking for backward compatibility: room_url, token = await configure(session)

    Raises:
        Exception: If DAILY_API_KEY is not provided in environment variables.

    Examples::

        # Standard room
        room_url, token = await configure(session)

        # SIP-enabled room
        sip_config = await configure(session, sip_caller_phone="+15551234567")
        print(f"SIP endpoint: {sip_config.sip_endpoint}")

        # Custom room properties with recording enabled
        custom_props = DailyRoomProperties(
            enable_recording="cloud",
            max_participants=2,
        )
        config = await configure(session, room_properties=custom_props)
    """
    # Check for required API key
    api_key = os.getenv("DAILY_API_KEY")
    if not api_key:
        raise Exception(
            "DAILY_API_KEY environment variable is required. "
            "Get your API key from https://dashboard.daily.co/developers"
        )

    # Warn if both room_properties and individual parameters are provided
    if room_properties is not None:
        individual_params_provided = any(
            [
                room_exp_duration != 2.0,
                token_exp_duration != 2.0,
                sip_caller_phone is not None,
                sip_enable_video is not False,
                sip_num_endpoints != 1,
                sip_codecs is not None,
            ]
        )
        if individual_params_provided:
            logger.warning(
                "Both room_properties and individual parameters (room_exp_duration, token_exp_duration, "
                "sip_*) were provided. The room_properties will be used and individual parameters "
                "will be ignored."
            )

    # Determine if SIP mode is enabled
    sip_enabled = sip_caller_phone is not None

    # If room_properties is provided, check if it has SIP configuration
    if room_properties and room_properties.sip:
        sip_enabled = True

    daily_rest_helper = DailyRESTHelper(
        daily_api_key=api_key,
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )

    # Check for existing room URL (only in standard mode)
    existing_room_url = os.getenv("DAILY_SAMPLE_ROOM_URL")
    if existing_room_url and not sip_enabled:
        # Use existing room (standard mode only)
        logger.info(f"Using existing Daily room: {existing_room_url}")
        room_url = existing_room_url

        # Create token and return standard format
        expiry_time: float = token_exp_duration * 60 * 60
        token = await daily_rest_helper.get_token(room_url, expiry_time)
        return DailyRoomConfig(room_url=room_url, token=token)

    # Create a new room
    room_prefix = "pipecat-sip" if sip_enabled else "pipecat"
    room_name = f"{room_prefix}-{uuid.uuid4().hex[:8]}"
    logger.info(f"Creating new Daily room: {room_name}")

    # Use provided room_properties or build from parameters
    if room_properties is None:
        # Calculate expiration time
        expiration_time = time.time() + (room_exp_duration * 60 * 60)

        # Create room properties
        room_properties = DailyRoomProperties(
            exp=expiration_time,
            eject_at_room_exp=True,
        )

        # Add SIP configuration if enabled
        if sip_enabled:
            sip_params = DailyRoomSipParams(
                display_name=sip_caller_phone,
                video=sip_enable_video,
                sip_mode="dial-in",
                num_endpoints=sip_num_endpoints,
                codecs=sip_codecs,
            )
            room_properties.sip = sip_params
            room_properties.enable_dialout = True  # Enable outbound calls if needed
            room_properties.start_video_off = not sip_enable_video  # Voice-only by default

    # Create room parameters
    room_params = DailyRoomParams(name=room_name, properties=room_properties)

    try:
        room_response = await daily_rest_helper.create_room(room_params)
        room_url = room_response.url
        logger.info(f"Created Daily room: {room_url}")

        # Create meeting token
        token_expiry_seconds = token_exp_duration * 60 * 60
        token = await daily_rest_helper.get_token(room_url, token_expiry_seconds)

        if sip_enabled:
            # Return SIP configuration object
            sip_endpoint = room_response.config.sip_endpoint
            logger.info(f"SIP endpoint: {sip_endpoint}")

            return DailyRoomConfig(
                room_url=room_url,
                token=token,
                sip_endpoint=sip_endpoint,
            )
        else:
            # Return standard configuration
            return DailyRoomConfig(room_url=room_url, token=token)

    except Exception as e:
        error_msg = f"Error creating Daily room: {e}"
        logger.error(error_msg)
        raise


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

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(
            "configure_with_args is deprecated. Use configure() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    room_url, token = await configure(aiohttp_session)
    return (room_url, token, None)
