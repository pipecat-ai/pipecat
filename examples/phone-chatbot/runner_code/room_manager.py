import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from fastapi import HTTPException

from pipecat.transports.services.helpers.daily_rest import (
    DailyRESTHelper,
    DailyRoomObject,
    DailyRoomParams,
    DailyRoomProperties,
    DailyRoomSipParams,
)


class RoomType(Enum):
    VOICEMAIL_DETECTION = "voicemail_detection"
    DIALOUT_WITH_VOICEMAIL = "dialout_with_voicemail"
    USER_DIALIN = "user_dialin"
    OPERATOR_ROOM = "operator_room"


@dataclass
class RoomRequest:
    detect_voicemail: bool = False
    dialout_number: Optional[str] = None
    operator_number: Optional[str] = None
    call_id: Optional[str] = None
    call_domain: Optional[str] = None
    from_number: Optional[str] = None
    to_number: Optional[str] = None

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "RoomRequest":
        return cls(
            detect_voicemail=data.get("detectVoicemail", False),
            dialout_number=data.get("dialoutNumber"),
            operator_number=data.get("operatorNumber"),
            call_id=data.get("callId"),
            call_domain=data.get("callDomain"),
            from_number=data.get("From"),
            to_number=data.get("To"),
        )


class RoomManager:
    def __init__(self, daily_helpers: Dict[str, DailyRESTHelper], room_url: Optional[str] = None):
        self.daily_helpers = daily_helpers
        self.room_url = room_url
        self.operator_number = "+12097844759"  # Consider moving this to config
        self.max_session_time = 5 * 60  # 5 minutes

    def determine_room_type(self, request: RoomRequest) -> RoomType:
        """Determine the type of room based on the request parameters."""
        if request.to_number:
            if request.to_number == self.operator_number:
                return RoomType.OPERATOR_ROOM
            return RoomType.USER_DIALIN
        elif request.dialout_number:
            return RoomType.DIALOUT_WITH_VOICEMAIL
        else:
            return RoomType.VOICEMAIL_DETECTION

    async def create_room(self, request: RoomRequest) -> Dict[str, Any]:
        """Create a room based on the request type."""
        room_type = self.determine_room_type(request)

        if room_type == RoomType.OPERATOR_ROOM:
            room = await self._create_operator_room()
            result = {
                "room": room,
                "room_url": room.url,
                "sipUri": room.config.sip_endpoint,
                "is_operator_room": True,  # Flag to indicate this is an operator room
            }
        else:
            room = await self._create_standard_room(request, room_type)
            result = {
                "room": room,
                "room_url": room.url,
                "sipUri": room.config.sip_endpoint,
                "is_operator_room": False,
            }

        # Get token for the room
        token = await self.daily_helpers["rest"].get_token(room.url, self.max_session_time)
        if not token:
            raise HTTPException(status_code=500, detail="Failed to get room token")

        # Only spawn bot for non-operator rooms
        if room_type != RoomType.OPERATOR_ROOM:
            await self._spawn_bot(room=room, token=token, request=request, room_type=room_type)
        return result

    async def _create_operator_room(self) -> DailyRoomObject:
        """Create a room specifically for operator handling."""
        properties = DailyRoomProperties(
            sip=DailyRoomSipParams(
                display_name="operator-user", video=False, sip_mode="dial-in", num_endpoints=1
            )
        )
        return await self._create_or_get_room(properties)

    async def _create_standard_room(
        self, request: RoomRequest, room_type: RoomType
    ) -> DailyRoomObject:
        """Create a standard room with appropriate properties."""
        properties = DailyRoomProperties(
            sip=DailyRoomSipParams(
                display_name="dialin-user", video=False, sip_mode="dial-in", num_endpoints=1
            )
        )

        # Enable dialout if needed
        if room_type in [
            RoomType.DIALOUT_WITH_VOICEMAIL,
            RoomType.VOICEMAIL_DETECTION,
            RoomType.USER_DIALIN,
        ]:
            properties.enable_dialout = True

        return await self._create_or_get_room(properties)

    async def _create_or_get_room(self, properties: DailyRoomProperties) -> DailyRoomObject:
        """Create a new room or get existing one based on URL."""
        if not self.room_url:
            params = DailyRoomParams(properties=properties)
            return await self.daily_helpers["rest"].create_room(params=params)

        try:
            room = await self.daily_helpers["rest"].get_room_from_url(self.room_url)
            return room
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to get room {self.room_url}: {str(e)}"
            )

    async def _spawn_bot(
        self, room: DailyRoomObject, token: str, request: RoomRequest, room_type: RoomType
    ) -> None:
        """Spawn the appropriate bot for the room."""
        # Check required parameters
        if not room.url or not token:
            raise HTTPException(
                status_code=500,
                detail="Missing required parameters: room URL and token are required",
            )

        # For dialin or voicemail detection, we need call_id and call_domain
        if room_type in [RoomType.USER_DIALIN]:
            if not request.call_id or not request.call_domain:
                raise HTTPException(
                    status_code=500,
                    detail="Missing required parameters: call_id and call_domain are required for dialin",
                )

        # Get the parent directory (where bot_daily.py is located)
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Gets runner_code directory
        parent_dir = os.path.dirname(current_dir)  # Gets the directory containing bot_daily.py

        # Construct command with required arguments
        bot_cmd = [
            "python3",
            os.path.join(parent_dir, "bot_daily.py"),
            "-u",
            room.url,
            "-t",
            token,
        ]

        # Add optional arguments
        if request.call_id:
            bot_cmd.extend(["-i", request.call_id])
        if request.call_domain:
            bot_cmd.extend(["-d", request.call_domain])
        if request.detect_voicemail:
            bot_cmd.append("-v")
        if request.dialout_number:
            bot_cmd.extend(["-o", request.dialout_number])
        if request.operator_number:
            bot_cmd.extend(["-op", request.operator_number])

        try:
            subprocess.Popen(bot_cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to start bot: {str(e)}")
