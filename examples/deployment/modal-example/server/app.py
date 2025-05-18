"""modal_example.

This module shows a simple example of how to deploy a bot using Modal and FastAPI.

It includes:
- FastAPI endpoints for starting agents and checking bot statuses.
- Dynamic loading of bot implementations.
- Use of a Daily transport for bot communication.
"""

#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import importlib
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Literal

import aiohttp
import modal
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

# container specifications for the FastAPI web server
web_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("pipecat-ai[daily]")
    .add_local_dir("src", remote_path="/root/src")
)

# container specifications for the Pipecat pipeline
bot_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("pipecat-ai[daily,elevenlabs,openai,silero,google]")
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App("pipecat-modal", secrets=[modal.Secret.from_dotenv()])

router = APIRouter()

bot_jobs = {}
daily_helpers = {}

# Names of all supported bot implementations
# These correspond to the bot files in the src directory
BotName = Literal["openai", "gemini", "vllm"]


def cleanup():
    """Cleanup function to terminate all bot processes.

    Called during server shutdown.
    """
    for entry in bot_jobs.values():
        func = modal.FunctionCall.from_id(entry[0])
        if func:
            func.cancel()


def get_bot_file(bot_name: BotName) -> str:
    """Retrieve the bot file name corresponding to the provided bot_name.

    Args:
        bot_name (BotName): The name of the bot (e.g., 'openai', 'gemini', 'vllm').

    Returns:
        str: The file name corresponding to the bot implementation.

    Raises:
        ValueError: If the bot name is invalid or not supported.
    """
    # bot_implementation = os.getenv("BOT_IMPLEMENTATION", "openai").lower().strip()
    bot_implementation = bot_name.lower().strip()
    if not bot_implementation:
        bot_implementation = "openai"
    if bot_implementation not in ["openai", "gemini", "vllm"]:
        raise ValueError(
            f"Invalid BOT_IMPLEMENTATION: {bot_implementation}. Must be 'openai' or 'gemini' or 'vllm'"
        )

    return f"bot_{bot_implementation}"


def get_runner(path: str, bot_file: str) -> callable:
    """Dynamically import the run_bot function based on the bot name.

    Args:
        path (str): The path to the bot files (e.g., 'src').
        bot_file (str): The file name of the bot implementation (e.g., 'openai', 'gemini', 'vllm').

    Returns:
        function: The run_bot function from the specified bot module.

    Raises:
        ImportError: If the specified bot module or run_bot function is not found.
    """
    try:
        # Dynamically construct the module name
        module_name = f"{path}.{bot_file}"
        # Import the module
        module = importlib.import_module(module_name)
        # Get the run_bot function from the module
        return getattr(module, "run_bot")
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import run_bot from {module_name}: {e}")


async def create_room_and_token() -> tuple[str, str]:
    """Create a Daily room and generate an authentication token.

    This function checks for existing room URL and token in the environment variables.
    If not found, it creates a new room using the Daily API and generates a token for it.

    Returns:
        tuple[str, str]: A tuple containing the room URL and the authentication token.

    Raises:
        HTTPException: If room creation or token generation fails.
    """
    from pipecat.transports.services.helpers.daily_rest import DailyRoomParams

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


@app.function(image=bot_image, min_containers=1)
async def bot_runner(room_url, token, bot_name: BotName = "openai"):
    """Launch the provided bot process, providing the given room URL and token for the bot to join.

    Args:
        room_url (str): The URL of the Daily room where the bot and client will communicate.
        token (str): The authentication token for the room.
        bot_name (BotName): The name of the bot implementation to use. Defaults to "openai".

    Raises:
        HTTPException: If the bot pipeline fails to start.
    """
    try:
        path = "src"
        bot_file = get_bot_file(bot_name)
        run_bot = get_runner(path, bot_file)

        print(f"Starting bot process: {bot_file} -u {room_url} -t {token}")
        await run_bot(room_url, token)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start bot pipeline: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager that handles startup and shutdown tasks.

    - Creates aiohttp session
    - Initializes Daily API helper
    - Cleans up resources on shutdown
    """
    from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper

    aiohttp_session = aiohttp.ClientSession()
    daily_helpers["rest"] = DailyRESTHelper(
        daily_api_key=os.getenv("DAILY_API_KEY", ""),
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )
    yield
    await aiohttp_session.close()
    cleanup()


class ConnectData(BaseModel):
    """Data provided by client to specify the bot pipeline.

    Attributes:
        bot_name (BotName): The name of the bot to connect to. Defaults to "openai".
    """

    bot_name: BotName = "openai"


async def start(data: ConnectData):
    """Internal method to start a bot agent and return the room URL and token.

    Args:
        data (ConnectData): The data containing the bot name to use.

    Returns:
        tuple[str, str]: A tuple containing the room URL and token.
    """
    room_url, token = await create_room_and_token()
    launch_bot_func = modal.Function.from_name("pipecat-modal", "bot_runner")
    function_id = launch_bot_func.spawn(room_url, token, data.bot_name)
    bot_jobs[function_id] = (function_id, room_url)

    return room_url, token


@router.get("/")
async def start_agent():
    """A user endpoint for launching a bot agent and redirecting to the created room URL.

    This function retrieves the bot implementation from the environment,
    starts the bot agent, and redirects the user to the room URL to
    interact with the bot through a Daily Prebuilt Interface.

    Returns:
        RedirectResponse: A response that redirects to the room URL.
    """
    bot_name = os.getenv("BOT_IMPLEMENTATION", "openai").lower().strip()
    print(f"Starting bot: {bot_name}")
    room_url, token = await start(ConnectData(bot_name=bot_name))

    return RedirectResponse(room_url)


@router.post("/connect")
async def rtvi_connect(data: ConnectData) -> Dict[Any, Any]:
    """A user endpoint for launching a bot agent and retrieving the room/token credentials.

    This function retrieves the bot implementation from the request, if provided,
    starts the bot agent, and returns the room URL and token for the bot. This allows the
    client to then connect to the bot using their own RTVI interface.

    Args:
        data (ConnectData): Optional. The data containing the bot name to use.

    Returns:
        Dict[Any, Any]: A dictionary containing the room URL and token.
    """
    print(f"Starting bot: {data.bot_name}")
    if data is None or not data.bot_name:
        data.bot_name = os.getenv("BOT_IMPLEMENTATION", "openai").lower().strip()
    room_url, token = await start(data)

    return {"room_url": room_url, "token": token}


@router.get("/status/{fid}")
def get_status(fid: str):
    """Retrieve the status of a bot process by its function ID.

    Args:
        fid (str): The function ID of the bot process.

    Returns:
        JSONResponse: A JSON response containing the bot's status and result code.

    Raises:
        HTTPException: If the bot process with the given ID is not found.
    """
    func = modal.FunctionCall.from_id(fid)
    if not func:
        raise HTTPException(status_code=404, detail=f"Bot with process id: {fid} not found")

    try:
        result = func.get(timeout=0)
        return JSONResponse({"bot_id": fid, "status": "finished", "code": result})
    except modal.exception.OutputExpiredError:
        return JSONResponse({"bot_id": fid, "status": "finished", "code": 404})
    except TimeoutError:
        return JSONResponse({"bot_id": fid, "status": "running", "code": 202})


@app.function(image=web_image, min_containers=1)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def fastapi_app():
    """Create and configure the FastAPI application.

    This function initializes the FastAPI app with middleware, routes, and lifespan management.
    It is decorated to be used as a Modal ASGI app.
    """
    from fastapi.middleware.cors import CORSMiddleware

    # Initialize FastAPI app
    web_app = FastAPI(lifespan=lifespan)

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include the endpoints from endpoints.py
    web_app.include_router(router)

    return web_app
