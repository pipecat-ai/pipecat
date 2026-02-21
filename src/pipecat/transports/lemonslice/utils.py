#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LemonSlice API utilities for session management.

This module provides helper classes for interacting with the LemonSlice API,
including session creation and termination.
"""

from typing import Any, Optional

import aiohttp
from loguru import logger


class LemonSliceApi:
    """Helper class for interacting with the LemonSlice API.

    Provides methods for creating and managing sessions with LemonSlice avatars.
    """

    LEMONSLICE_URL = "https://lemonslice.com/api/liveai/sessions"

    def __init__(self, api_key: str, session: aiohttp.ClientSession):
        """Initialize the LemonSliceApi client.

        Args:
            api_key: LemonSlice API key for authentication.
            session: An aiohttp session for making HTTP requests.
        """
        self._api_key = api_key
        self._session = session
        self._headers = {"Content-Type": "application/json", "x-api-key": self._api_key}

    async def create_session(
        self,
        *,
        agent_image_url: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_prompt: Optional[str] = None,
        idle_timeout: Optional[int] = None,
        daily_room_url: Optional[str] = None,
        daily_token: Optional[str] = None,
        properties: Optional[dict[str, Any]] = None,
    ) -> dict:
        """Create a new session with the specified agent_id or agent_image_url.

        Args:
            agent_image_url: The URL to an agent image. Provide either agent_id or agent_image_url.
            agent_id: ID of a LemonSlice agent. Provide either agent_id or agent_image_url.
            agent_prompt: A high-level system prompt that subtly influences the avatar’s movements, expressions, and emotional demeanor.
            idle_timeout: Idle timeout in seconds.
            daily_room_url: Daily room URL to use for the session.
            daily_token: Daily token for authenticating with the room.
            properties: Additional properties to pass to the session.

        Returns:
            Dictionary containing session_id, room_url, and control_url.

        Raises:
            ValueError: If neither agent_id nor agent_image_url is provided.
        """
        if not agent_id and not agent_image_url:
            raise ValueError("Provide either agent_id or agent_image_url")
        if agent_id and agent_image_url:
            raise ValueError("Provide exactly one of agent_id or agent_image_url, not both")

        logger.debug(
            f"Creating LemonSlice session: agent_id={agent_id}, agent_image_url={agent_image_url}"
        )
        payload: dict[str, object] = {"transport_type": "daily"}
        if agent_id is not None:
            payload["agent_id"] = agent_id
        if agent_image_url is not None:
            payload["agent_image_url"] = agent_image_url
        if agent_prompt is not None:
            payload["agent_prompt"] = agent_prompt
        if idle_timeout is not None:
            payload["idle_timeout"] = idle_timeout
        properties_dict: dict[str, Any] = dict(properties) if properties else {}
        if daily_room_url is not None:
            properties_dict["daily_url"] = daily_room_url
        if daily_token is not None:
            properties_dict["daily_token"] = daily_token
        if properties_dict:
            payload["properties"] = properties_dict
        async with self._session.post(
            self.LEMONSLICE_URL, headers=self._headers, json=payload
        ) as r:
            r.raise_for_status()
            response = await r.json()
            logger.debug(f"Created LemonSlice session: {response}")
            return response

    async def end_session(self, session_id: str, control_url: str):
        """End an existing session.

        Args:
            session_id: ID of the session to end.
            control_url: The control URL from the create_session response.
        """
        payload = {"event": "terminate"}
        async with self._session.post(control_url, headers=self._headers, json=payload) as r:
            r.raise_for_status()
            logger.debug(f"Ended LemonSlice session {session_id}")
