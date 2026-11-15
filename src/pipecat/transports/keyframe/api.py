#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Keyframe API utilities for session management.

This module provides a helper class for interacting with the Keyframe API to
create avatar sessions in Daily rooms.
"""

from typing import Any

import aiohttp
from loguru import logger


class KeyframeApi:
    """Helper class for interacting with the Keyframe API.

    Provides a method for creating sessions with Keyframe avatars. A session
    ends when the bot leaves its Daily room.
    """

    KEYFRAME_URL = "https://api.keyframelabs.com"

    def __init__(self, api_key: str, session: aiohttp.ClientSession):
        """Initialize the KeyframeApi client.

        Args:
            api_key: Keyframe API key for authentication.
            session: An aiohttp session for making HTTP requests.
        """
        self._api_key = api_key
        self._session = session

    async def create_session(
        self,
        *,
        source_participant_name: str,
        persona_id: str | None = None,
        persona_slug: str | None = None,
        daily_room_url: str | None = None,
        avatar_daily_token: str | None = None,
        source_participant_id: str | None = None,
        api_url: str | None = None,
    ) -> dict[str, Any]:
        """Create a new session with the specified persona_id or persona_slug.

        Args:
            source_participant_name: User name the bot joins the Daily room with.
                The avatar speaks the audio that participant sends it.
            persona_id: ID of the Keyframe persona.
            persona_slug: Slug of the Keyframe persona, such as ``public:cosmo_persona-2.0-live``.
            daily_room_url: Daily room URL to use for the session. When None,
                Keyframe creates a private Daily room.
            avatar_daily_token: Daily token for the avatar to join
                ``daily_room_url`` with.
            source_participant_id: The ``user_id`` of the Daily token the bot
                joins ``daily_room_url`` with. The avatar then knows the bot by
                that ID instead of by its user name.
            api_url: Keyframe API URL override.

        Returns:
            Dictionary containing reservation_id, model_id, protocol,
            avatar_participant_name and room_url, plus avatar_participant_id,
            bot_token and user_token when Keyframe created the room.

        Raises:
            ValueError: If zero or both of persona_id and persona_slug are provided.
        """
        if (persona_id is None) == (persona_slug is None):
            raise ValueError("Provide exactly one of persona_id or persona_slug")

        logger.debug(
            f"Creating Keyframe session: persona_id={persona_id}, persona_slug={persona_slug}"
        )
        payload: dict[str, Any] = {"source_participant_name": source_participant_name}
        if persona_id is not None:
            payload["persona_id"] = persona_id
        if persona_slug is not None:
            payload["persona_slug"] = persona_slug
        if daily_room_url is not None:
            payload["room_url"] = daily_room_url
        if avatar_daily_token is not None:
            payload["avatar_token"] = avatar_daily_token
        if source_participant_id is not None:
            payload["source_participant_id"] = source_participant_id

        url = f"{api_url or self.KEYFRAME_URL}/v1/sessions/plugins/daily"
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        async with self._session.post(url, headers=headers, json=payload) as r:
            if not r.ok:
                # The body says why: the plan, the persona's model, or the room.
                logger.error(f"Keyframe refused the session ({r.status}): {await r.text()}")
            r.raise_for_status()
            response = await r.json()
            logger.debug(f"Created Keyframe session {response.get('reservation_id')}")
            return response
