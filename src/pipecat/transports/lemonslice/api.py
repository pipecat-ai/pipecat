#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LemonSlice API utilities for session management.

This module provides helper classes for interacting with the LemonSlice API,
including session creation and termination.
"""

import io
import json
from typing import Any

import aiohttp
from loguru import logger
from PIL import Image


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
        agent_image_url: str | None = None,
        agent_id: str | None = None,
        agent_image: Image.Image | None = None,
        agent_prompt: str | None = None,
        idle_timeout: int | None = None,
        daily_room_url: str | None = None,
        daily_token: str | None = None,
        connection_properties: dict[str, Any] | None = None,
        extra_properties: dict[str, Any] | None = None,
        api_url: str | None = None,
    ) -> dict:
        """Create a new session with the specified agent_id, agent_image_url, or agent_image.

        Args:
            agent_image_url: URL to an agent image.
            agent_id: ID of a LemonSlice agent.
            agent_image: PIL image of the agent.
            agent_prompt: A high-level system prompt that subtly influences the avatar's
                movements, expressions, and emotional demeanor.
            idle_timeout: Idle timeout in seconds.
            daily_room_url: Daily room URL to use for the session.
            daily_token: Daily token for authenticating with the room.
            connection_properties: Additional connection properties to pass to the session.
            extra_properties: Additional top-level keys to merge into the payload.
            api_url: LemonSlice API URL override.

        Returns:
            Dictionary containing session_id, room_url, and control_url.

        Raises:
            ValueError: If zero or more than one of agent_id, agent_image_url, or agent_image
                is provided.
        """
        given_sources = [
            source for source in (agent_id, agent_image_url, agent_image) if source is not None
        ]
        if len(given_sources) == 0:
            raise ValueError("Provide exactly one of agent_id, agent_image_url, or agent_image")
        if len(given_sources) > 1:
            raise ValueError(
                "Provide exactly one of agent_id, agent_image_url, or agent_image, not multiple"
            )

        logger.debug(
            f"Creating LemonSlice session: agent_id={agent_id}, "
            f"agent_image_url={agent_image_url}, agent_image={'set' if agent_image else None}"
        )
        payload: dict[str, Any] = {"transport_type": "daily"}
        if agent_id is not None:
            payload["agent_id"] = agent_id
        if agent_image_url is not None:
            payload["agent_image_url"] = agent_image_url
        if agent_prompt is not None:
            payload["agent_prompt"] = agent_prompt
        if idle_timeout is not None:
            payload["idle_timeout"] = idle_timeout
        properties_dict: dict[str, Any] = (
            dict(connection_properties) if connection_properties else {}
        )
        if daily_room_url is not None:
            properties_dict["daily_url"] = daily_room_url
        if daily_token is not None:
            properties_dict["daily_token"] = daily_token
        if properties_dict:
            payload["properties"] = properties_dict
        if extra_properties:
            payload.update(extra_properties)

        image_bytes: bytes | None = None
        if agent_image is not None:
            image_bytes = _encode_image(agent_image)

        url = api_url if api_url is not None else self.LEMONSLICE_URL
        response = await self._post(url, payload, image_bytes=image_bytes)
        logger.debug(f"Created LemonSlice session: {response}")
        return response

    async def _post(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        image_bytes: bytes | None = None,
    ) -> dict:
        """POST to the LemonSlice API as JSON or multipart form data.

        Args:
            url: Request URL.
            payload: JSON payload for the session request.
            image_bytes: Optional PNG-encoded image.

        Returns:
            Parsed JSON response body.
        """
        headers = {"x-api-key": self._api_key}
        if image_bytes is not None:
            form = aiohttp.FormData()
            form.add_field("payload", json.dumps(payload), content_type="application/json")
            form.add_field(
                "image",
                image_bytes,
                filename="image.png",
                content_type="image/png",
            )
            async with self._session.post(url, headers=headers, data=form) as r:
                r.raise_for_status()
                return await r.json()

        headers["Content-Type"] = "application/json"
        async with self._session.post(url, headers=headers, json=payload) as r:
            r.raise_for_status()
            return await r.json()

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


def _encode_image(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes for a multipart upload."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
