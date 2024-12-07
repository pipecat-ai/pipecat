#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import subprocess
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomParams

MAX_BOTS_PER_ROOM = 1

# Bot sub-process dict for status reporting and concurrency control
bot_procs = {}
daily_helpers = {}


def cleanup():
    # Clean up function, just to be extra safe
    for entry in bot_procs.values():
        proc = entry[0]
        proc.terminate()
        proc.wait()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Verify required environment variables
    if not os.getenv("DAILY_API_KEY"):
        raise Exception("DAILY_API_KEY environment variable is required")

    aiohttp_session = aiohttp.ClientSession()
    daily_helpers["rest"] = DailyRESTHelper(
        daily_api_key=os.getenv("DAILY_API_KEY"),
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )
    yield
    await aiohttp_session.close()
    cleanup()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def start_agent(request: Request):
    print("!!! Creating room")
    room = await daily_helpers["rest"].create_room(DailyRoomParams())
    print(f"!!! Room URL: {room.url}")

    if not room.url:
        raise HTTPException(
            status_code=500,
            detail="Missing 'room' property in request data. Cannot start agent without a target room!",
        )

    # Check if there is already an existing process running in this room
    num_bots_in_room = sum(
        1 for proc in bot_procs.values() if proc[1] == room.url and proc[0].poll() is None
    )
    if num_bots_in_room >= MAX_BOTS_PER_ROOM:
        raise HTTPException(status_code=500, detail=f"Max bot limited reach for room: {room.url}")

    # Get the token for the room
    token = await daily_helpers["rest"].get_token(room.url)

    if not token:
        raise HTTPException(status_code=500, detail=f"Failed to get token for room: {room.url}")

    # Spawn a new agent and join the user session
    try:
        # Modified to use environment variables instead of command line arguments
        env = os.environ.copy()
        env["DAILY_ROOM_URL"] = room.url
        env["DAILY_ROOM_TOKEN"] = token

        proc = subprocess.Popen(
            ["python3", "-m", "bot"],
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        bot_procs[proc.pid] = (proc, room.url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")

    return RedirectResponse(room.url)


@app.get("/status/{pid}")
def get_status(pid: int):
    # Look up the subprocess
    proc = bot_procs.get(pid)

    # If the subprocess doesn't exist, return an error
    if not proc:
        raise HTTPException(status_code=404, detail=f"Bot with process id: {pid} not found")

    # Check the status of the subprocess
    if proc[0].poll() is None:
        status = "running"
    else:
        status = "finished"

    return JSONResponse({"bot_id": pid, "status": status})


if __name__ == "__main__":
    import uvicorn

    # Get port from Heroku environment, default to 7860 if not available
    port = int(os.getenv("PORT", "7860"))

    # In Heroku, we must bind to 0.0.0.0
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload on Heroku
    )
