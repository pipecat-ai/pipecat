import asyncio
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


@dataclass
class JoinPayload:
    room: str
    session_id: str
    participant_id: str
    participant_name: Optional[str]


class WebhookHandler:
    def __init__(self, room_state_manager, daily_domain: str = "bdom.daily.co"):
        self.room_state_manager = room_state_manager
        self.daily_domain = daily_domain

    def _build_room_url(self, room_name: str) -> str:
        """Build the full room URL from a room name."""
        return f"https://{self.daily_domain}/{room_name}"

    def _parse_join_payload(self, data: dict) -> JoinPayload:
        """Parse and validate the join webhook payload."""
        payload = data.get("payload", {})
        if not payload.get("room"):
            raise HTTPException(status_code=400, detail="Missing 'room' in payload")

        return JoinPayload(
            room=payload["room"],
            session_id=payload.get("session_id", ""),
            participant_id=payload.get("participant_id", ""),
            participant_name=payload.get("participant_name"),
        )

    async def handle_join_webhook(self, request: Request) -> JSONResponse:
        """Handle the joined_room webhook."""
        print("Processing participant join webhook")
        try:
            data = await request.json()
            if "test" in data:
                return JSONResponse({"test": True})

            # Parse and validate payload
            payload = self._parse_join_payload(data)
            room_url = self._build_room_url(payload.room)

            # Store joined room state
            self.room_state_manager.store_joined_room(room_url)

            # Wait briefly to ensure all systems are ready
            await asyncio.sleep(5)

            # Handle the room join event
            await self.room_state_manager.handle_room_join(room_url)

            return JSONResponse(
                {
                    "message": "Joined room event processed",
                    "room_url": room_url,
                    "session_id": payload.session_id,
                }
            )

        except Exception as e:
            print(f"Error handling join webhook: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to process room join webhook: {str(e)}"
            )
