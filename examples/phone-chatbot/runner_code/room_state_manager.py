import asyncio
from dataclasses import dataclass
from typing import Dict, Optional

import httpx
from fastapi import HTTPException

from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomObject


@dataclass
class RoomState:
    """Represents the state of a room including call information."""

    room_url: str
    room_object: Optional[DailyRoomObject] = None
    call_id: Optional[str] = None
    call_domain: Optional[str] = None
    sip_uri: Optional[str] = None


class RoomStateManager:
    def __init__(self, daily_api_key: str, daily_api_url: str, daily_helpers: Dict):
        self.daily_api_key = daily_api_key
        self.daily_api_url = daily_api_url
        self.daily_helpers = daily_helpers
        self._created_rooms: Dict[str, RoomState] = {}
        self._joined_rooms: Dict[str, RoomState] = {}
        self._updated_rooms: set[str] = set()
        self._operator_room_url: Optional[str] = None  # Track operator room specifically

    def store_created_room(
        self,
        room: DailyRoomObject,
        call_id: Optional[str] = None,
        call_domain: Optional[str] = None,
        is_operator_room: bool = False,
    ) -> None:
        """Store information about a newly created room."""
        # Extract the correct SIP URI format
        sip_uri = None
        if hasattr(room.config, "sip_uri") and isinstance(room.config.sip_uri, dict):
            endpoint = room.config.sip_uri.get("endpoint")
            if endpoint:
                sip_uri = f"sip:{endpoint}"

        self._created_rooms[room.url] = RoomState(
            room_url=room.url,
            room_object=room,
            call_id=call_id,
            call_domain=call_domain,
            sip_uri=sip_uri,
        )

        if is_operator_room:
            self._operator_room_url = room.url
            print(f"+++++ Stored operator room URL: {room.url}")

    def store_joined_room(self, room_url: str) -> None:
        """Store information about a room that has been joined."""
        self._joined_rooms[room_url] = RoomState(room_url=room_url)

    def get_created_room(self, room_url: str) -> Optional[RoomState]:
        """Get information about a created room."""
        return self._created_rooms.get(room_url)

    def get_joined_room(self, room_url: str) -> Optional[RoomState]:
        """Get information about a joined room."""
        return self._joined_rooms.get(room_url)

    def is_room_updated(self, room_url: str) -> bool:
        """Check if a room has already been updated."""
        return room_url in self._updated_rooms

    def mark_room_updated(self, room_url: str) -> None:
        """Mark a room as having been updated."""
        self._updated_rooms.add(room_url)

    def get_operator_room_url(self) -> Optional[str]:
        """Get the URL of the operator room if it exists."""
        return self._operator_room_url

    async def handle_room_join(self, room_url: str) -> None:
        """Handle the event of a participant joining a room."""
        # If we have an operator room and it matches the join event, use that
        if self._operator_room_url == room_url:
            # This is the operator room join
            created_room = self._created_rooms[room_url]
        else:
            # This is the non-operator room join
            print(f"Non-operator room join for URL: {room_url}")
            if room_url not in self._created_rooms:
                print(f"Room {room_url} not found in created rooms")
                return
            created_room = self._created_rooms[room_url]

        if self.is_room_updated(room_url):
            print("Room already updated, skipping...")
            return

        if not created_room.call_id or not created_room.call_domain or not created_room.sip_uri:
            print("Missing required room data for update")
            return

        await self._send_dialin_update(created_room)
        self.mark_room_updated(room_url)

        # Now send app message to the non-operator room (we're using room_url here
        # because we want to send to the original room, not the operator room)
        if room_url != self._operator_room_url:  # Only send if this is NOT the operator room
            # Extract room name from URL (e.g., get 'hello' from 'https://bdom.daily.co/hello')
            room_name = room_url.split("/")[-1]

            app_message_url = f"{self.daily_api_url}/rooms/{room_name}/send-app-message"
            headers = {
                "Authorization": f"Bearer {self.daily_api_key}",
                "Content-Type": "application/json",
            }
            payload = {"data": {"test": 1}, "recipient": "*"}

            print(f"+++++ Sending app message to non-operator room: {room_name}")
            async with httpx.AsyncClient() as client:
                response = await client.post(app_message_url, headers=headers, json=payload)

            if response.status_code != 200:
                print(f"Error sending app message: {response.status_code}, {response.text}")
                raise HTTPException(status_code=response.status_code, detail=response.text)

            print("+++++ App message sent successfully!")

    async def _send_dialin_update(self, room: RoomState) -> None:
        """Send an API request to update the dial-in status."""
        print(f"Sending dial-in update for room: {room.room_url}")

        dialin_url = f"{self.daily_api_url}/dialin/pinlessCallUpdate"
        headers = {
            "Authorization": f"Bearer {self.daily_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "callId": room.call_id,
            "callDomain": room.call_domain,
            "sipUri": room.sip_uri,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(dialin_url, headers=headers, json=payload)

        if response.status_code != 200:
            print(f"Error from Daily API: {response.status_code}, {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)
