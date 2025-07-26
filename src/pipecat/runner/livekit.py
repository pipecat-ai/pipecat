#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LiveKit room and token configuration utilities.

This module provides helper functions for creating and configuring LiveKit
rooms and authentication tokens. It handles JWT token generation with
appropriate grants for both regular participants and AI agents.

The module supports creating tokens for development and testing, with
automatic agent detection for proper room permissions.

Required environment variables:

- LIVEKIT_API_KEY - LiveKit API key
- LIVEKIT_API_SECRET - LiveKit API secret
- LIVEKIT_URL - LiveKit server URL
- LIVEKIT_ROOM_NAME - Room name to join

Example::

    from pipecat.runner.livekit import configure

    url, token, room_name = await configure()
    # Use with LiveKitTransport
"""

import argparse
import os
from typing import Optional

from livekit import api
from loguru import logger


def generate_token(room_name: str, participant_name: str, api_key: str, api_secret: str) -> str:
    """Generate a LiveKit access token for a participant.

    Args:
        room_name: Name of the LiveKit room.
        participant_name: Name of the participant.
        api_key: LiveKit API key.
        api_secret: LiveKit API secret.

    Returns:
        JWT token string for room access.
    """
    token = api.AccessToken(api_key, api_secret)
    token.with_identity(participant_name).with_name(participant_name).with_grants(
        api.VideoGrants(
            room_join=True,
            room=room_name,
        )
    )

    return token.to_jwt()


def generate_token_with_agent(
    room_name: str, participant_name: str, api_key: str, api_secret: str
) -> str:
    """Generate a LiveKit access token for an agent participant.

    Args:
        room_name: Name of the LiveKit room.
        participant_name: Name of the participant.
        api_key: LiveKit API key.
        api_secret: LiveKit API secret.

    Returns:
        JWT token string for agent room access.
    """
    token = api.AccessToken(api_key, api_secret)
    token.with_identity(participant_name).with_name(participant_name).with_grants(
        api.VideoGrants(
            room_join=True,
            room=room_name,
            agent=True,  # This makes LiveKit client know agent has joined
        )
    )

    return token.to_jwt()


async def configure():
    """Configure LiveKit room URL and token from arguments or environment.

    Returns:
        Tuple containing the server URL, authentication token, and room name.

    Raises:
        Exception: If required LiveKit configuration is not provided.
    """
    (url, token, room_name, _) = await configure_with_args()
    return (url, token, room_name)


async def configure_with_args(parser: Optional[argparse.ArgumentParser] = None):
    """Configure LiveKit room with command-line argument parsing.

    Args:
        parser: Optional argument parser. If None, creates a default one.

    Returns:
        Tuple containing server URL, authentication token, room name, and parsed arguments.

    Raises:
        Exception: If required LiveKit configuration is not provided via arguments or environment.
    """
    if not parser:
        parser = argparse.ArgumentParser(description="LiveKit AI SDK Bot Sample")
    parser.add_argument(
        "-r", "--room", type=str, required=False, help="Name of the LiveKit room to join"
    )
    parser.add_argument("-u", "--url", type=str, required=False, help="URL of the LiveKit server")

    args, unknown = parser.parse_known_args()

    room_name = args.room or os.getenv("LIVEKIT_ROOM_NAME")
    url = args.url or os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not room_name:
        raise Exception(
            "No LiveKit room specified. Use the -r/--room option from the command line, or set LIVEKIT_ROOM_NAME in your environment."
        )

    if not url:
        raise Exception(
            "No LiveKit server URL specified. Use the -u/--url option from the command line, or set LIVEKIT_URL in your environment."
        )

    if not api_key or not api_secret:
        raise Exception(
            "LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set in environment variables."
        )

    token = generate_token_with_agent(room_name, "Pipecat Agent", api_key, api_secret)

    # Generate user token for testing/debugging
    user_token = generate_token(room_name, "User", api_key, api_secret)
    logger.info(f"User token: {user_token}")

    return (url, token, room_name, args)
