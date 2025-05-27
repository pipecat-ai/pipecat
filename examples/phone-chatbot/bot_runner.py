import argparse
import json
import os
import shlex
import subprocess
from contextlib import asynccontextmanager
from typing import Any, Dict

import aiohttp
from bot_constants import (
    MAX_SESSION_TIME,
    REQUIRED_ENV_VARS,
)
from bot_definitions import bot_registry
from bot_runner_helpers import (
    determine_room_capabilities,
    ensure_prompt_config,
    process_dialin_request,
)
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pipecat.transports.services.helpers.daily_rest import (
    DailyRESTHelper,
    DailyRoomParams,
    DailyRoomProperties,
    DailyRoomSipParams,
)

load_dotenv(override=True)

daily_helpers = {}


# ----------------- Daily Room Management ----------------- #


async def create_daily_room(room_url: str = None, config_body: Dict[str, Any] = None):
    """Create or retrieve a Daily room with appropriate properties based on the configuration.

    Args:
        room_url: Optional existing room URL
        config_body: Optional configuration that determines room capabilities

    Returns:
        Dict containing room URL, token, and SIP endpoint
    """
    if not room_url:
        # Get room capabilities based on the configuration
        capabilities = determine_room_capabilities(config_body)

        # Configure SIP parameters if dialin is needed
        sip_params = None
        if capabilities["enable_dialin"]:
            sip_params = DailyRoomSipParams(
                display_name="dialin-user", video=False, sip_mode="dial-in", num_endpoints=2
            )

        # Create the properties object with the appropriate settings
        properties = DailyRoomProperties(sip=sip_params)

        # Set dialout capability if needed
        if capabilities["enable_dialout"]:
            properties.enable_dialout = True

        # Log the capabilities being used
        capability_str = ", ".join([f"{k}={v}" for k, v in capabilities.items()])
        print(f"Creating room with capabilities: {capability_str}")

        params = DailyRoomParams(properties=properties)

        print("Creating new room...")
        room = await daily_helpers["rest"].create_room(params=params)
    else:
        # Check if passed room URL exists
        try:
            room = await daily_helpers["rest"].get_room_from_url(room_url)
        except Exception:
            raise HTTPException(status_code=500, detail=f"Room not found: {room_url}")

    print(f"Daily room: {room.url} {room.config.sip_endpoint}")

    # Get token for the agent
    token = await daily_helpers["rest"].get_token(room.url, MAX_SESSION_TIME)

    if not room or not token:
        raise HTTPException(status_code=500, detail="Failed to get room or token")

    return {"room": room.url, "token": token, "sip_endpoint": room.config.sip_endpoint}


# ----------------- Bot Process Management ----------------- #


async def start_bot(room_details: Dict[str, str], body: Dict[str, Any], example: str) -> bool:
    """Start a bot process with the given configuration.

    Args:
        room_details: Room URL and token
        body: Bot configuration
        example: Example script to run

    Returns:
        Boolean indicating success
    """
    room_url = room_details["room"]
    token = room_details["token"]

    # Properly format body as JSON string for command line
    body_json = json.dumps(body).replace('"', '\\"')
    print(f"++++ Body JSON: {body_json}")

    # Modified to use non-LLM-specific bot module names
    bot_proc = f'python3 -m {example} -u {room_url} -t {token} -b "{body_json}"'
    print(f"Starting bot. Example: {example}, Room: {room_url}")

    try:
        command_parts = shlex.split(bot_proc)
        subprocess.Popen(command_parts, bufsize=1, cwd=os.path.dirname(os.path.abspath(__file__)))
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")


# ----------------- API Setup ----------------- #


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


# ----------------- API Endpoints ----------------- #


@app.post("/start")
async def handle_start_request(request: Request) -> JSONResponse:
    """Unified endpoint to handle bot configuration for different scenarios."""
    # Get default room URL from environment
    room_url = os.getenv("DAILY_SAMPLE_ROOM_URL", None)

    try:
        data = await request.json()

        # Handle webhook test
        if "test" in data:
            return JSONResponse({"test": True})

        # Handle direct dialin webhook from Daily
        if all(key in data for key in ["From", "To", "callId", "callDomain"]):
            body = await process_dialin_request(data)
        # Handle body-based request
        elif "config" in data:
            # Use the registry to set up the bot configuration
            body = bot_registry.setup_configuration(data["config"])
        else:
            raise HTTPException(status_code=400, detail="Invalid request format")

        # Ensure prompt configuration
        body = ensure_prompt_config(body)

        # Detect which bot type to use
        bot_type_name = bot_registry.detect_bot_type(body)
        if not bot_type_name:
            raise HTTPException(
                status_code=400, detail="Configuration doesn't match any supported scenario"
            )

        # Create the Daily room
        room_details = await create_daily_room(room_url, body)

        # Start the bot
        await start_bot(room_details, body, bot_type_name)

        # Get the bot type
        bot_type = bot_registry.get_bot(bot_type_name)

        # Build the response
        response = {"status": "Bot started", "bot_type": bot_type_name}

        # Add room URL for test mode
        if bot_type.has_test_mode(body):
            response["room_url"] = room_details["room"]
            # Remove llm_model from response as it's no longer relevant
            if "llm" in body:
                response["llm_provider"] = body["llm"]  # Optionally keep track of provider

        # Add dialout info for dialout scenarios
        if "dialout_settings" in body and len(body["dialout_settings"]) > 0:
            first_setting = body["dialout_settings"][0]
            if "phoneNumber" in first_setting:
                response["dialing_to"] = f"phone:{first_setting['phoneNumber']}"
            elif "sipUri" in first_setting:
                response["dialing_to"] = f"sip:{first_setting['sipUri']}"

        return JSONResponse(response)

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Request processing error: {str(e)}")


# ----------------- Main ----------------- #

if __name__ == "__main__":
    # Check environment variables
    for env_var in REQUIRED_ENV_VARS:
        if env_var not in os.environ:
            raise Exception(f"Missing environment variable: {env_var}.")

    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument(
        "--host", type=str, default=os.getenv("HOST", "0.0.0.0"), help="Host address"
    )
    parser.add_argument("--port", type=int, default=os.getenv("PORT", 7860), help="Port number")
    parser.add_argument("--reload", action="store_true", default=True, help="Reload code on change")

    config = parser.parse_args()

    try:
        import uvicorn

        uvicorn.run("bot_runner:app", host=config.host, port=config.port, reload=config.reload)

    except KeyboardInterrupt:
        print("Pipecat runner shutting down...")
