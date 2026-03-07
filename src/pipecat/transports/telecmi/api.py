import asyncio
import inspect
import json
import logging
import signal
from typing import Awaitable, Callable, Dict, Optional

import socketio

logger = logging.getLogger(__name__)

DEFAULT_SIGNALING_URL = "https://signaling.piopiy.com"

class TelecmiAPI:
    """
    Handles Socket.IO signaling with TeleCMI to receive incoming calls 
    and spawn Pipecat sessions dynamically.
    """
    def __init__(
        self,
        agent_id: str,
        agent_token: str,
        create_session: Callable[..., Awaitable[None]],
        signaling_url: Optional[str] = None,
        debug: bool = False,
    ):
        """
        create_session(url, token, room_name, **kwargs) -> coroutine
        """
        self.signaling_url = signaling_url or DEFAULT_SIGNALING_URL
        self.agent_id = agent_id
        self.agent_token = agent_token
        self.create_session = create_session
        self.debug = debug

        self.sio = socketio.AsyncClient(logger=self.debug, engineio_logger=self.debug)
        self.active_sessions: Dict[str, asyncio.Task] = {}

        self._setup_events()

    def _setup_events(self) -> None:
        @self.sio.event
        async def connect():
            logger.info(f"Connected to TeleCMI signaling as agent {self.agent_id}")

        @self.sio.on("join_room")
        async def handle_join_session(invite: dict):
            room = invite.get("room_name")
            token = invite.get("token")
            url = invite.get("url") or self.signaling_url

            if not room or not token:
                logger.warning(f"Invalid join_room payload: {invite}")
                return

            existing = self.active_sessions.get(room)
            if existing and not existing.done():
                logger.warning(f"Session {room} already running")
                return
            
            async def session_runner():
                try:
                    sig = inspect.signature(self.create_session)
                    kwargs = {}
                    
                    # Pass the standard WebRTC/Transport connection variables
                    if "room_name" in sig.parameters: kwargs["room_name"] = room
                    if "token" in sig.parameters: kwargs["token"] = token
                    if "url" in sig.parameters: kwargs["url"] = url
                    
                    # Pass additional TeleCMI metadata
                    if "call_id" in sig.parameters: kwargs["call_id"] = invite.get("call_id")
                    if "agent_id" in sig.parameters: kwargs["agent_id"] = invite.get("agent_id")
                    if "from_number" in sig.parameters: kwargs["from_number"] = invite.get("from_number")
                    if "to_number" in sig.parameters: kwargs["to_number"] = invite.get("to_number")
                    
                    if "metadata" in sig.parameters:
                        raw_meta = invite.get("metadata")
                        parsed_meta = raw_meta
                        if raw_meta and isinstance(raw_meta, str):
                            try:
                                parsed_meta = json.loads(raw_meta.replace(r"\}", "}"))
                            except (TypeError, json.JSONDecodeError):
                                pass
                        kwargs["metadata"] = parsed_meta

                    if self.debug:
                        logger.debug(f"session_runner starting with kwargs: {kwargs}")

                    await self.create_session(**kwargs)
                except Exception as e:
                    logger.error(f"Error in TeleCMI session {room}: {e}", exc_info=True)

            task = asyncio.create_task(
                session_runner(),
                name=f"telecmi_session:{room}",
            )

            self.active_sessions[room] = task
            task.add_done_callback(lambda _: self.active_sessions.pop(room, None))

        @self.sio.on("cancel_room")
        async def handle_cancel_session(data: dict):
            room = data.get("room_name")
            if not room:
                return
            task = self.active_sessions.pop(room, None)
            if task and not task.done():
                logger.info(f"Cancelling session {room} by server request")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def connect(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            except NotImplementedError:
                pass # Windows fallback
        
        await self.sio.connect(
            self.signaling_url,
            auth={"agent_id": self.agent_id, "token": self.agent_token},
        )
        try:
            await self.sio.wait()
        except asyncio.CancelledError:
            pass

    async def shutdown(self) -> None:
        logger.info("Shutting down TeleCMI agent signaling...")
        try:
            await self.sio.disconnect()
        except Exception:
            pass

        tasks = list(self.active_sessions.values())
        self.active_sessions.clear()
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("TeleCMI Agent signaling shutdown complete.")
