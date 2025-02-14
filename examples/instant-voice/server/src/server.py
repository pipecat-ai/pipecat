#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomParams

# Load environment variables
load_dotenv(override=True)

NUMBER_OF_ROOMS = 1


class RoomPool:
    """Manages a pool of pre-created rooms for quick allocation."""

    def __init__(self, daily_rest_helper: DailyRESTHelper):
        self.daily_rest_helper = daily_rest_helper
        self.pool: List[Dict[str, str]] = []
        self.lock = asyncio.Lock()

    async def fill_pool(self, count: int):
        """Fills the pool with `count` new rooms."""
        for _ in range(count):
            await self.add_room()

    async def add_room(self):
        """Creates a new room and adds it to the pool."""
        try:
            room = await self.daily_rest_helper.create_room(DailyRoomParams())
            if not room.url:
                raise HTTPException(status_code=500, detail="Failed to create room")

            user_token = await self.daily_rest_helper.get_token(room.url)
            if not user_token:
                raise HTTPException(status_code=500, detail="Failed to get user token")

            bot_token = await self.daily_rest_helper.get_token(room.url)
            if not bot_token:
                raise HTTPException(status_code=500, detail="Failed to get bot token")

            async with self.lock:
                self.pool.append(
                    {"room_url": room.url, "user_token": user_token, "bot_token": bot_token}
                )

        except Exception as e:
            print(f"Error adding room to pool: {e}")

    async def get_room(self) -> Dict[str, str]:
        """Retrieves a room from the pool and requests a new one to maintain the size."""
        async with self.lock:
            if not self.pool:
                raise HTTPException(status_code=503, detail="No available rooms")

            room = self.pool.pop(0)  # Get first available room

        # Start a background task to replenish the pool
        asyncio.create_task(self.add_room())

        return room

    async def delete_room(self, room_url: str):
        """Deletes a room when it is not needed anymore"""
        await self.daily_rest_helper.delete_room_by_url(room_url)

    async def cleanup(self):
        for rooms in self.pool:
            room_url = rooms["room_url"]
            await self.delete_room(room_url)


class BotManager:
    """Manages bot subprocesses asynchronously."""

    def __init__(self):
        self.bot_procs: Dict[int, asyncio.subprocess.Process] = {}
        self.room_mappings: Dict[int, str] = {}  # Maps process ID to room URL

    async def start_bot(self, room_url: str, token: str) -> int:
        bot_file = "single_bot"
        command = f"python3 -m {bot_file} -u {room_url} -t {token}"

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            if proc.pid is None:
                raise HTTPException(status_code=500, detail="Failed to get subprocess PID")

            self.bot_procs[proc.pid] = proc
            self.room_mappings[proc.pid] = room_url
            # Monitor the process and delete the room when it exits
            asyncio.create_task(self._monitor_process(proc.pid))

            return proc.pid
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")

    async def _monitor_process(self, pid: int):
        """Monitors a bot process and deletes the associated room when it exits."""
        proc = self.bot_procs.get(pid)
        if proc:
            await proc.wait()  # Wait for the process to exit
            room_url = self.room_mappings.pop(pid, None)

            if room_url:
                await room_pool.delete_room(room_url)
                print(f"Deleted room: {room_url}")

            del self.bot_procs[pid]

    async def cleanup(self):
        """Terminates all running bot processes and deletes associated rooms."""
        for pid, proc in list(self.bot_procs.items()):
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5)

                room_url = self.room_mappings.pop(pid, None)
                if room_url:
                    await room_pool.delete_room(room_url)  # Delete room when process terminates
                    print(f"Deleted room: {room_url}")

            except asyncio.TimeoutError:
                print(f"Process {pid} did not terminate in time.")
            except Exception as e:
                print(f"Error terminating process {pid}: {e}")

        # Clear remaining mappings
        self.bot_procs.clear()
        self.room_mappings.clear()


# Global instances
bot_manager = BotManager()
room_pool: RoomPool  # Will be initialized in lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles FastAPI startup and shutdown."""
    global room_pool
    aiohttp_session = aiohttp.ClientSession()
    daily_rest_helper = DailyRESTHelper(
        daily_api_key=os.getenv("DAILY_API_KEY", ""),
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )

    room_pool = RoomPool(daily_rest_helper)
    await room_pool.fill_pool(NUMBER_OF_ROOMS)  # Fill pool on startup

    yield  # Run app

    await bot_manager.cleanup()
    await room_pool.cleanup()
    await aiohttp_session.close()


# Initialize FastAPI app with lifespan manager
app = FastAPI(lifespan=lifespan)

# Configure CORS to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/connect")
async def bot_connect(request: Request) -> Dict[Any, Any]:
    try:
        room = await room_pool.get_room()
        await bot_manager.start_bot(room["room_url"], room["bot_token"])
    except HTTPException as e:
        return {"error": str(e)}

    return {
        "room_url": room["room_url"],
        "token": room["user_token"],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
