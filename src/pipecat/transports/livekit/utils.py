#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LiveKit REST Helpers.

Methods that wrap the LiveKit API for room management.
"""

import aiohttp


class LiveKitRESTHelper:
    """Helper class for interacting with LiveKit's REST API.

    Provides methods for managing LiveKit rooms.
    """

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        api_url: str = "https://your-livekit-host.com",
        aiohttp_session: aiohttp.ClientSession,
    ):
        """Initialize the LiveKit REST helper.

        Args:
            api_key: Your LiveKit API key.
            api_secret: Your LiveKit API secret.
            api_url: LiveKit server URL (e.g. "https://your-livekit-host.com").
            aiohttp_session: Async HTTP session for making requests.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_url = api_url.rstrip("/")
        self.aiohttp_session = aiohttp_session

    def _create_access_token(self, room_create: bool = True) -> str:
        """Create a signed access token for LiveKit API authentication.

        Args:
            room_create: Whether to grant roomCreate permission.

        Returns:
            Signed JWT access token.
        """
        import time

        import jwt

        claims = {
            "iss": self.api_key,
            "sub": self.api_key,
            "nbf": int(time.time()),
            "exp": int(time.time()) + 60,  # Token valid for 60 seconds
            "video": {
                "roomCreate": room_create,
            },
        }

        return jwt.encode(claims, self.api_secret, algorithm="HS256")

    async def delete_room_by_name(self, room_name: str) -> bool:
        """Delete a LiveKit room by name.

        This will forcibly disconnect all participants currently in the room.

        Args:
            room_name: Name of the room to delete.

        Returns:
            True if deletion was successful.

        Raises:
            Exception: If deletion fails.
        """
        token = self._create_access_token(room_create=True)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        async with self.aiohttp_session.post(
            f"{self.api_url}/twirp/livekit.RoomService/DeleteRoom",
            headers=headers,
            json={"room": room_name},
        ) as r:
            if r.status != 200:
                text = await r.text()
                raise Exception(f"Failed to delete room [{room_name}] (status: {r.status}): {text}")

        return True
