"""
bot_runner.py

HTTP service that listens for incoming calls from either Daily or Twilio,
provisioning a room and starting a Pipecat bot in response.

Refer to README for more information.
"""


import aiohttp
import os
import argparse
import subprocess
import uuid
import sys

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse

from pipecat.transports.services.helpers.daily_rest import (
    DailyRESTHelper,
    DailyRoomObject,
    DailyRoomProperties,
    DailyRoomSipParams,
    DailyRoomParams)

from dotenv import load_dotenv
load_dotenv(override=True)


# ------------ Configuration ------------ #

MAX_SESSION_TIME = 5 * 60  # 5 minutes
REQUIRED_ENV_VARS = ['OPENAI_API_KEY', 'DAILY_API_KEY',
                     'ELEVENLABS_API_KEY', 'ELEVENLABS_VOICE_ID']

daily_helpers = {}

# ----------------- API ----------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    aiohttp_session = aiohttp.ClientSession()
    daily_helpers["rest"] = DailyRESTHelper(
        daily_api_key=os.getenv("DAILY_API_KEY", ""),
        daily_api_url=os.getenv("DAILY_API_URL", 'https://api.daily.co/v1'),
        aiohttp_session=aiohttp_session
    )
    yield
    await aiohttp_session.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

"""
Create Daily room, tell the bot if the room is created for Twilio's SIP or Daily's SIP (vendor).
When the vendor is Daily, the bot handles the call forwarding automatically,
i.e, forwards the call from the "hold music state" to the Daily Room's SIP URI.

Alternatively, when the vendor is Twilio (not Daily), the bot is responsible for
updating the state on Twilio. So when `dialin-ready` fires, it takes appropriate
action using the Twilio Client library.
"""


async def _create_daily_room(room_url, callId, phone_number, callDomain=None):

    if not room_url:
        params = DailyRoomParams(
            properties=DailyRoomProperties(
                # Note: these are the default values, except for the display name
                sip=DailyRoomSipParams(
                    display_name="dialin-user",
                    video=False,
                    sip_mode="dial-in",
                    num_endpoints=1,
                ),
                enable_dialout=True,
            )
        )

        print("Creating new room...")
        room: DailyRoomObject = await daily_helpers["rest"].create_room(params=params)

    else:
        # Check passed room URL exist (we assume that it already has a sip set up!)
        try:
            print(f"Joining existing room: {room_url}")
            room: DailyRoomObject = await daily_helpers["rest"].get_room_from_url(room_url)
        except Exception:
            raise HTTPException(
                status_code=500, detail=f"Room not found: {room_url}")

    # Give the agent a token to join the session
    token = await daily_helpers["rest"].get_token(room.url, MAX_SESSION_TIME)

    if not room or not token:
        raise HTTPException(
            status_code=500, detail="Failed to get room or token token")

    # Spawn a new agent, and join the user session
    # Note: this is mostly for demonstration purposes (refer to 'deployment' in docs)
    bot_proc = f"{sys.executable} -m bot_daily -u {room.url} -t {token} -i {callId} -d {callDomain} -p {phone_number}"

    try:
        subprocess.Popen(
            [bot_proc],
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start subprocess: {e}")

    return room


@app.post("/outbound_call")
async def outbound_call(request: Request) -> JSONResponse:
    # Get the phone number from the request body
    data = await request.json()
    phone_number = data.get("phone_number")

    if not phone_number:
        raise HTTPException(status_code=400, detail="Phone number is required")

    # create callId
    callId = str(uuid.uuid4())

    await _create_daily_room(
        room_url=None,
        callId=callId,
        phone_number=phone_number,
        callDomain="b1ec7700-e975-47f0-8704-8ef4880a2486",
    )

    return JSONResponse({"success": True})

# ----------------- Main ----------------- #


if __name__ == "__main__":
    # Check environment variables
    for env_var in REQUIRED_ENV_VARS:
        if env_var not in os.environ:
            raise Exception(f"Missing environment variable: {env_var}.")

    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument("--host", type=str,
                        default=os.getenv("HOST", "0.0.0.0"), help="Host address")
    parser.add_argument("--port", type=int,
                        default=os.getenv("PORT", 7860), help="Port number")
    parser.add_argument("--reload", action="store_true",
                        default=True, help="Reload code on change")

    config = parser.parse_args()

    try:
        import uvicorn

        uvicorn.run(
            "bot_runner:app",
            host=config.host,
            port=config.port,
            reload=config.reload
        )

    except KeyboardInterrupt:
        print("Pipecat runner shutting down...")
