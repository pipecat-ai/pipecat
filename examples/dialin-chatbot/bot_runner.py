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
    DailyRoomParams,
)

from dotenv import load_dotenv

load_dotenv(override=True)


# ------------ Configuration ------------ #

MAX_SESSION_TIME = 5 * 60  # 5 minutes
REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "DAILY_API_KEY", "ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"]

daily_helpers = {}

# ----------------- API ----------------- #


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

"""
Create Daily room, tell the bot if the room is created for Twilio's SIP or Daily's SIP (vendor).
When the vendor is Daily, the bot handles the call forwarding automatically,
i.e, forwards the call from the "hold music state" to the Daily Room's SIP URI.

Alternatively, when the vendor is Twilio (not Daily), the bot is responsible for
updating the state on Twilio. So when `dialin-ready` fires, it takes appropriate
action using the Twilio Client library.
"""


async def _create_daily_room(room_url, callId, callDomain=None, vendor="daily"):
    if not room_url:
        params = DailyRoomParams(
            properties=DailyRoomProperties(
                # Note: these are the default values, except for the display name
                sip=DailyRoomSipParams(
                    display_name="dialin-user", video=False, sip_mode="dial-in", num_endpoints=1
                )
            )
        )

        print(f"Creating new room...")
        room: DailyRoomObject = await daily_helpers["rest"].create_room(params=params)

    else:
        # Check passed room URL exist (we assume that it already has a sip set up!)
        try:
            print(f"Joining existing room: {room_url}")
            room: DailyRoomObject = await daily_helpers["rest"].get_room_from_url(room_url)
        except Exception:
            raise HTTPException(status_code=500, detail=f"Room not found: {room_url}")

    print(f"Daily room: {room.url} {room.config.sip_endpoint}")

    # Give the agent a token to join the session
    token = await daily_helpers["rest"].get_token(room.url, MAX_SESSION_TIME)

    if not room or not token:
        raise HTTPException(status_code=500, detail=f"Failed to get room or token token")

    # Spawn a new agent, and join the user session
    # Note: this is mostly for demonstration purposes (refer to 'deployment' in docs)
    if vendor == "daily":
        bot_proc = f"python3 -m bot_daily -u {room.url} -t {token} -i {callId} -d {callDomain}"
    else:
        bot_proc = f"python3 -m bot_twilio -u {room.url} -t {token} -i {callId} -s {room.config.sip_endpoint}"

    try:
        subprocess.Popen(
            [bot_proc], shell=True, bufsize=1, cwd=os.path.dirname(os.path.abspath(__file__))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")

    return room


@app.post("/twilio_start_bot", response_class=PlainTextResponse)
async def twilio_start_bot(request: Request):
    print(f"POST /twilio_voice_bot")

    # twilio_start_bot is invoked directly by Twilio (as a web hook).
    # On Twilio, under Active Numbers, pick the phone number
    # Click Configure and under Voice Configuration,
    # "a call comes in" choose webhook and point the URL to
    # where this code is hosted.
    data = {}
    try:
        # shouldnt have received json, twilio sends form data
        form_data = await request.form()
        data = dict(form_data)
    except Exception:
        pass

    room_url = os.getenv("DAILY_SAMPLE_ROOM_URL", None)
    callId = data.get("CallSid")

    if not callId:
        raise HTTPException(status_code=500, detail="Missing 'CallSid' in request")

    print("CallId: %s" % callId)

    # create room and tell the bot to join the created room
    # note: Twilio does not require a callDomain
    room: DailyRoomObject = await _create_daily_room(room_url, callId, None, "twilio")

    print(f"Put Twilio on hold...")
    # We have the room and the SIP URI,
    # but we do not know if the Daily SIP Worker and the Bot have joined the call
    # put the call on hold until the 'on_dialin_ready' fires.
    # Then, the bot will update the called sid with the sip uri.
    # http://com.twilio.music.classical.s3.amazonaws.com/BusyStrings.mp3
    resp = VoiceResponse()
    resp.play(
        url="http://com.twilio.sounds.music.s3.amazonaws.com/MARKOVICHAMP-Borghestral.mp3", loop=10
    )
    return str(resp)


@app.post("/daily_start_bot")
async def daily_start_bot(request: Request) -> JSONResponse:
    # The /daily_start_bot is invoked when a call is received on Daily's SIP URI
    # daily_start_bot will create the room, put the call on hold until
    # the bot and sip worker are ready. Daily will automatically
    # forward the call to the SIP URi when dialin_ready fires.

    # Use specified room URL, or create a new one if not specified
    room_url = os.getenv("DAILY_SAMPLE_ROOM_URL", None)
    # Get the dial-in properties from the request
    try:
        data = await request.json()
        if "test" in data:
            # Pass through any webhook checks
            return JSONResponse({"test": True})
        callId = data.get("callId", None)
        callDomain = data.get("callDomain", None)
    except Exception:
        raise HTTPException(status_code=500, detail="Missing properties 'callId' or 'callDomain'")

    print(f"CallId: {callId}, CallDomain: {callDomain}")
    room: DailyRoomObject = await _create_daily_room(room_url, callId, callDomain, "daily")

    # Grab a token for the user to join with
    return JSONResponse({"room_url": room.url, "sipUri": room.config.sip_endpoint})


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
