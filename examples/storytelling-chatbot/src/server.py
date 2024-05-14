import os
import argparse
import subprocess
import atexit
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from utils.daily_helpers import create_room as _create_room, get_token, get_name_from_url

MAX_BOTS_PER_ROOM = 1

# Bot sub-process dict for status reporting and concurrency control
bot_procs = {}


def cleanup():
    # Clean up function, just to be extra safe
    for proc in bot_procs.values():
        proc.terminate()
        proc.wait()


atexit.register(cleanup)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory
STATIC_DIR = "frontend/out"

app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")


@app.post("/create")
async def create_room(request: Request) -> JSONResponse:
    data = await request.json()

    if data.get('room_url') is not None:
        room_url = data.get('room_url')
        room_name = get_name_from_url(room_url)
    else:
        room_url, room_name = _create_room()

    token = get_token(room_url)

    return JSONResponse({"room_url": room_url, "room_name": room_name, "token": token})


@app.post("/start")
async def start_agent(request: Request) -> JSONResponse:
    data = await request.json()

    # Is this a webhook creation request?
    if "test" in data:
        return JSONResponse({"test": True})

    # Ensure the room property is present
    room_url = data.get('room_url')
    if not room_url:
        raise HTTPException(
            status_code=500,
            detail="Missing 'room' property in request data. Cannot start agent without a target room!")

    # Check if there is already an existing process running in this room
    num_bots_in_room = sum(
        1 for proc in bot_procs.values() if proc[1] == room_url and proc[0].poll() is None)
    if num_bots_in_room >= MAX_BOTS_PER_ROOM:
        raise HTTPException(
            status_code=500, detail=f"Max bot limited reach for room: {room_url}")

    # Get the token for the room
    token = get_token(room_url)

    if not token:
        raise HTTPException(
            status_code=500, detail=f"Failed to get token for room: {room_url}")

    # Spawn a new agent, and join the user session
    # Note: this is mostly for demonstration purposes (refer to 'deployment' in README)
    try:
        proc = subprocess.Popen(
            [
                f"python3 -m bot -u {room_url} -t {token}"
            ],
            shell=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        bot_procs[proc.pid] = (proc, room_url)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start subprocess: {e}")

    return JSONResponse({"bot_id": proc.pid, "room_url": room_url})


@app.get("/status/{pid}")
def get_status(pid: int):
    # Look up the subprocess
    proc = bot_procs.get(pid)

    # If the subprocess doesn't exist, return an error
    if not proc:
        raise HTTPException(
            status_code=404, detail=f"Bot with process id: {pid} not found")

    # Check the status of the subprocess
    if proc[0].poll() is None:
        status = "running"
    else:
        status = "finished"

    return JSONResponse({"bot_id": pid, "status": status})


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


if __name__ == "__main__":
    # Check environment variables
    required_env_vars = ['OPENAI_API_KEY', 'DAILY_API_KEY',
                         'FAL_KEY', 'ELEVENLABS_VOICE_ID', 'ELEVENLABS_API_KEY']
    for env_var in required_env_vars:
        if env_var not in os.environ:
            raise Exception(f"Missing environment variable: {env_var}.")

    import uvicorn

    default_host = os.getenv("HOST", "0.0.0.0")
    default_port = int(os.getenv("FAST_API_PORT", "7860"))

    parser = argparse.ArgumentParser(
        description="Daily Storyteller FastAPI server")
    parser.add_argument("--host", type=str,
                        default=default_host, help="Host address")
    parser.add_argument("--port", type=int,
                        default=default_port, help="Port number")
    parser.add_argument("--reload", action="store_true",
                        help="Reload code on change")

    config = parser.parse_args()

    uvicorn.run(
        "server:app",
        host=config.host,
        port=config.port,
        reload=config.reload
    )
