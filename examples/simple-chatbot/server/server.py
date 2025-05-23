#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""RTVI Bot Server Implementation.

This FastAPI server manages RTVI bot instances and provides endpoints for both
direct browser access and RTVI client connections. It handles:
- Creating Daily rooms
- Managing bot processes
- Providing connection credentials
- Monitoring bot status

Requirements:
- Daily API key (set in .env file)
- Python 3.10+
- FastAPI
- Running bot implementation
"""

import argparse
import asyncio
import datetime
import json
import os
import subprocess
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomParams

# Load environment variables from .env file
load_dotenv(override=True)

# Maximum number of bot instances allowed per room
MAX_BOTS_PER_ROOM = 1

# Dictionary to track bot processes: {pid: (process, room_url)}
bot_procs = {}

# Global dictionaries to store session data
session_recordings = {}
session_timestamps = {}

# Create directories
os.makedirs("recordings", exist_ok=True)
os.makedirs("session_data", exist_ok=True)

# Store Daily API helpers
daily_helpers = {}


def cleanup():
    """Cleanup function to terminate all bot processes.

    Called during server shutdown.
    """
    for entry in bot_procs.values():
        proc = entry[0]
        proc.terminate()
        proc.wait()


def get_bot_file():
    """Get the bot implementation file name based on environment configuration.

    Returns:
        str: The bot implementation file name (e.g., 'bot-openai' or 'bot-gemini')

    Raises:
        ValueError: If BOT_IMPLEMENTATION env var has an invalid value
    """
    bot_implementation = os.getenv("BOT_IMPLEMENTATION", "openai").lower().strip()
    # If blank or None, default to openai
    if not bot_implementation:
        bot_implementation = "openai"
    if bot_implementation not in ["openai", "gemini"]:
        raise ValueError(
            f"Invalid BOT_IMPLEMENTATION: {bot_implementation}. Must be 'openai' or 'gemini'"
        )
    return f"bot-{bot_implementation}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager that handles startup and shutdown tasks.

    - Creates aiohttp session
    - Initializes Daily API helper
    - Cleans up resources on shutdown
    """
    # Load session data on startup
    load_session_data()

    aiohttp_session = aiohttp.ClientSession()
    daily_helpers["rest"] = DailyRESTHelper(
        daily_api_key=os.getenv("DAILY_API_KEY", ""),
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )
    yield
    await aiohttp_session.close()
    cleanup()


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

# Serve recordings directory
app.mount("/recordings", StaticFiles(directory="recordings"), name="recordings")


# API endpoint to get latest session
@app.get("/api/latest-session")
async def get_latest_session():
    """Get the most recent session ID from available sessions.

    Returns:
        dict: Contains either session_id or error message
    """
    print(f"API call: get_latest_session, available sessions: {list(session_timestamps.keys())}")
    if not session_timestamps:
        return {"error": "No sessions found yet. Please complete a call first."}

    # Get the most recent session ID by timestamp
    latest_session_id = max(session_timestamps.items(), key=lambda x: x[1])[0]
    print(f"Latest session ID: {latest_session_id}")
    return {"session_id": latest_session_id}


# API endpoint to get session recordings
@app.get("/api/recordings/{session_id}")
async def get_recordings(session_id: str):
    """Get recording files associated with a specific session.

    Args:
        session_id (str): The ID of the session to fetch recordings for

    Returns:
        dict: Contains recording file paths or error message
    """
    print(f"API call: get_recordings for session: {session_id}")
    print(f"Available sessions: {list(session_recordings.keys())}")
    if session_id in session_recordings:
        return session_recordings[session_id]
    return {"error": "Session not found"}


# API endpoint to manually save session data (can be called by bot)
@app.post("/api/sessions/{session_id}/recordings")
async def add_recording(session_id: str, speaker_type: str, filename: str):
    """Add a recording to a session. Called by the bot process."""
    update_session_data(session_id, speaker_type, filename)
    return {"status": "success"}


async def create_room_and_token() -> tuple[str, str]:
    """Helper function to create a Daily room and generate an access token.

    Returns:
        tuple[str, str]: A tuple containing (room_url, token)

    Raises:
        HTTPException: If room creation or token generation fails
    """
    room_url = os.getenv("DAILY_SAMPLE_ROOM_URL", None)
    token = os.getenv("DAILY_SAMPLE_ROOM_TOKEN", None)
    if not room_url:
        room = await daily_helpers["rest"].create_room(DailyRoomParams())
        if not room.url:
            raise HTTPException(status_code=500, detail="Failed to create room")
        room_url = room.url

        token = await daily_helpers["rest"].get_token(room_url)
        if not token:
            raise HTTPException(status_code=500, detail=f"Failed to get token for room: {room_url}")

    return room_url, token


@app.get("/")
async def start_agent(request: Request):
    """Endpoint for direct browser access to the bot.

    Creates a room, starts a bot instance, and redirects to the Daily room URL.

    Returns:
        RedirectResponse: Redirects to the Daily room URL

    Raises:
        HTTPException: If room creation, token generation, or bot startup fails
    """
    print("Creating room")
    room_url, token = await create_room_and_token()
    print(f"Room URL: {room_url}")

    # Check if there is already an existing process running in this room
    num_bots_in_room = sum(
        1 for proc in bot_procs.values() if proc[1] == room_url and proc[0].poll() is None
    )
    if num_bots_in_room >= MAX_BOTS_PER_ROOM:
        raise HTTPException(status_code=500, detail=f"Max bot limit reached for room: {room_url}")

    # Spawn a new bot process
    try:
        bot_file = get_bot_file()
        proc = subprocess.Popen(
            [f"python3 -m {bot_file} -u {room_url} -t {token}"],
            shell=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        bot_procs[proc.pid] = (proc, room_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")

    return RedirectResponse(room_url)


@app.post("/connect")
async def rtvi_connect(request: Request) -> Dict[Any, Any]:
    """RTVI connect endpoint that creates a room and returns connection credentials.

    This endpoint is called by RTVI clients to establish a connection.

    Returns:
        Dict[Any, Any]: Authentication bundle containing room_url and token

    Raises:
        HTTPException: If room creation, token generation, or bot startup fails
    """
    print("Creating room for RTVI connection")
    room_url, token = await create_room_and_token()
    print(f"Room URL: {room_url}")

    # Start the bot process
    try:
        bot_file = get_bot_file()
        proc = subprocess.Popen(
            [f"python3 -m {bot_file} -u {room_url} -t {token}"],
            shell=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        bot_procs[proc.pid] = (proc, room_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")

    # Return the authentication bundle in format expected by DailyTransport
    return {"room_url": room_url, "token": token}


@app.get("/status/{pid}")
def get_status(pid: int):
    """Get the status of a specific bot process.

    Args:
        pid (int): Process ID of the bot

    Returns:
        JSONResponse: Status information for the bot

    Raises:
        HTTPException: If the specified bot process is not found
    """
    # Look up the subprocess
    proc = bot_procs.get(pid)

    # If the subprocess doesn't exist, return an error
    if not proc:
        raise HTTPException(status_code=404, detail=f"Bot with process id: {pid} not found")

    # Check the status of the subprocess
    status = "running" if proc[0].poll() is None else "finished"
    return JSONResponse({"bot_id": pid, "status": status})


# Session data management functions
def save_session_data():
    """Save session data to disk."""
    try:
        # Save session timestamps
        with open("session_data/session_timestamps.json", "w") as f:
            timestamps_dict = {k: v.isoformat() for k, v in session_timestamps.items()}
            json.dump(timestamps_dict, f)

        # Save session recordings
        with open("session_data/session_recordings.json", "w") as f:
            json.dump(session_recordings, f)

        print(f"Session data saved to disk. Sessions: {list(session_timestamps.keys())}")
    except Exception as e:
        print(f"Error saving session data: {e}")


def load_session_data():
    """Load session data from disk."""
    global session_timestamps, session_recordings

    try:
        # Load session timestamps
        if os.path.exists("session_data/session_timestamps.json"):
            with open("session_data/session_timestamps.json", "r") as f:
                timestamps_dict = json.load(f)
                session_timestamps = {
                    k: datetime.datetime.fromisoformat(v) for k, v in timestamps_dict.items()
                }

        # Load session recordings
        if os.path.exists("session_data/session_recordings.json"):
            with open("session_data/session_recordings.json", "r") as f:
                session_recordings = json.load(f)

        print(f"Session data loaded from disk. Sessions: {list(session_timestamps.keys())}")
    except Exception as e:
        print(f"Error loading session data: {e}")
        session_timestamps = {}
        session_recordings = {}


# Function to update session data (called by bot)
def update_session_data(session_id: str, speaker_type: str, filename: str):
    """Update session recordings data. Called by the bot process."""
    global session_recordings, session_timestamps

    # Initialize session if it doesn't exist
    if session_id not in session_recordings:
        session_recordings[session_id] = {"user": [], "bot": [], "full": []}
        session_timestamps[session_id] = datetime.datetime.now()

    # Add the filename to the appropriate category
    session_recordings[session_id][speaker_type].append(filename)

    # Save to disk immediately
    save_session_data()


if __name__ == "__main__":
    import uvicorn

    # Parse command line arguments for server configuration
    default_host = os.getenv("HOST", "0.0.0.0")
    default_port = int(os.getenv("FAST_API_PORT", "7860"))

    parser = argparse.ArgumentParser(description="Daily Storyteller FastAPI server")
    parser.add_argument("--host", type=str, default=default_host, help="Host address")
    parser.add_argument("--port", type=int, default=default_port, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Reload code on change")

    config = parser.parse_args()

    # Start the FastAPI server
    uvicorn.run(
        "server:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
    )
