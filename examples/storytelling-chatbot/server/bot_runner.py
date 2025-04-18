#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pipecat.transports.services.helpers.daily_rest import (
    DailyRESTHelper,
    DailyRoomObject,
    DailyRoomParams,
    DailyRoomProperties,
)

load_dotenv(override=True)

# ------------ Fast API Config ------------ #

MAX_SESSION_TIME = 5 * 60  # 5 minutes

daily_helpers = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    aiohttp_session = aiohttp.ClientSession()
    daily_helpers["rest"] = DailyRESTHelper(
        daily_api_key=os.getenv("DAILY_API_KEY", ""),
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )
    yield
    await aiohttp_session.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory
STATIC_DIR = "client/out"


# ------------ Fast API Routes ------------ #

app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")


@app.post("/")
async def start_bot(request: Request) -> JSONResponse:
    if os.getenv("ENV", "dev") == "production":
        # Only allow requests from the specified domain
        host_header = request.headers.get("host")
        allowed_domains = ["storytelling-chatbot.fly.dev", "www.storytelling-chatbot.fly.dev"]
        # Check if the Host header matches the allowed domain
        if host_header not in allowed_domains:
            raise HTTPException(status_code=403, detail="Access denied")

    try:
        data = await request.json()
        # Is this a webhook creation request?
        if "test" in data:
            return JSONResponse({"test": True})
    except Exception as e:
        pass

    # Use specified room URL, or create a new one if not specified
    room_url = os.getenv("DAILY_SAMPLE_ROOM_URL", "")

    if not room_url:
        params = DailyRoomParams(properties=DailyRoomProperties())
        try:
            room: DailyRoomObject = await daily_helpers["rest"].create_room(params=params)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unable to provision room {e}")
    else:
        # Check passed room URL exists, we should assume that it already has a sip set up
        try:
            room: DailyRoomObject = await daily_helpers["rest"].get_room_from_url(room_url)
        except Exception:
            raise HTTPException(status_code=500, detail=f"Room not found: {room_url}")

    # Give the agent a token to join the session
    token = await daily_helpers["rest"].get_token(room.url, MAX_SESSION_TIME)

    if not room or not token:
        raise HTTPException(status_code=500, detail=f"Failed to get token for room: {room_url}")

    # Launch a new VM, or run as a shell process (not recommended)
    if os.getenv("RUN_AS_VM", False):
        try:
            await virtualize_bot(room.url, token)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to spawn VM: {e}")
    else:
        try:
            subprocess.Popen(
                [f"python -m bot -u {room.url} -t {token}"],
                shell=True,
                bufsize=1,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")

    # Grab a token for the user to join with
    user_token = await daily_helpers["rest"].get_token(room.url, MAX_SESSION_TIME)

    return JSONResponse(
        {
            "room_url": room.url,
            "token": user_token,
        }
    )


@app.get("/{path_name:path}", response_class=FileResponse)
async def catch_all(path_name: Optional[str] = ""):
    if path_name == "":
        return FileResponse(f"{STATIC_DIR}/index.html")

    file_path = Path(STATIC_DIR) / (path_name or "")

    if file_path.is_file():
        return file_path

    html_file_path = file_path.with_suffix(".html")
    if html_file_path.is_file():
        return FileResponse(html_file_path)

    raise HTTPException(status_code=450, detail="Incorrect API call")


# ------------ Virtualization ------------ #


async def virtualize_bot(room_url: str, token: str):
    """This is an example of how to virtualize the bot using Fly.io
    You can adapt this method to use whichever cloud provider you prefer.
    """
    FLY_API_HOST = os.getenv("FLY_API_HOST", "https://api.machines.dev/v1")
    FLY_APP_NAME = os.getenv("FLY_APP_NAME", "storytelling-chatbot")
    FLY_API_KEY = os.getenv("FLY_API_KEY", "")
    FLY_HEADERS = {"Authorization": f"Bearer {FLY_API_KEY}", "Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        # Use the same image as the bot runner
        async with session.get(
            f"{FLY_API_HOST}/apps/{FLY_APP_NAME}/machines", headers=FLY_HEADERS
        ) as r:
            if r.status != 200:
                text = await r.text()
                raise Exception(f"Unable to get machine info from Fly: {text}")

            data = await r.json()
            image = data[0]["config"]["image"]

        # Machine configuration
        cmd = f"python server/bot.py -u {room_url} -t {token}"
        cmd = cmd.split()
        worker_props = {
            "config": {
                "image": image,
                "auto_destroy": True,
                "init": {"cmd": cmd},
                "restart": {"policy": "no"},
                "guest": {"cpu_kind": "shared", "cpus": 1, "memory_mb": 512},
            },
        }

        # Spawn a new machine instance
        async with session.post(
            f"{FLY_API_HOST}/apps/{FLY_APP_NAME}/machines", headers=FLY_HEADERS, json=worker_props
        ) as r:
            if r.status != 200:
                text = await r.text()
                raise Exception(f"Problem starting a bot worker: {text}")

            data = await r.json()
            # Wait for the machine to enter the started state
            vm_id = data["id"]

        async with session.get(
            f"{FLY_API_HOST}/apps/{FLY_APP_NAME}/machines/{vm_id}/wait?state=started",
            headers=FLY_HEADERS,
        ) as r:
            if r.status != 200:
                text = await r.text()
                raise Exception(f"Bot was unable to enter started state: {text}")

        print(f"Machine joined room: {room_url}")


# ------------ Main ------------ #

if __name__ == "__main__":
    # Check environment variables
    required_env_vars = [
        "GOOGLE_API_KEY",
        "DAILY_API_KEY",
        "ELEVENLABS_VOICE_ID",
        "ELEVENLABS_API_KEY",
    ]
    for env_var in required_env_vars:
        if env_var not in os.environ:
            raise Exception(f"Missing environment variable: {env_var}.")

    import uvicorn

    default_host = os.getenv("HOST", "0.0.0.0")
    default_port = int(os.getenv("FAST_API_PORT", "7860"))

    parser = argparse.ArgumentParser(description="Daily Storyteller FastAPI server")
    parser.add_argument("--host", type=str, default=default_host, help="Host address")
    parser.add_argument("--port", type=int, default=default_port, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Reload code on change")

    config = parser.parse_args()

    uvicorn.run("bot_runner:app", host=config.host, port=config.port, reload=config.reload)
