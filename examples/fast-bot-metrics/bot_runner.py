"""
bot_runner.py

HTTP service that listens for incoming calls from either Daily or Twilio,
provisioning a room and starting a Pipecat bot in response.

Refer to README for more information.
"""
import os
import argparse
import subprocess

from pydantic import BaseModel, ValidationError
from typing import Optional

from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomObject, DailyRoomProperties, DailyRoomSipParams, DailyRoomParams

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse

from bot import BotSettings

from dotenv import load_dotenv
load_dotenv(override=True)


# ------------ Configuration ------------ #

MAX_SESSION_TIME = 5 * 60  # 5 minutes
REQUIRED_ENV_VARS = ['DAILY_API_URL', 'DAILY_API_KEY', 'DEEPGRAM_API_KEY']

daily_rest_helper = DailyRESTHelper(
    os.getenv("DAILY_API_KEY", ""),
    os.getenv("DAILY_API_URL", 'https://api.daily.co/v1'))


class RunnerSettings(BaseModel):
    prompt: Optional[str] = None
    deepgram_voice: Optional[str] = None
    openai_model: Optional[str] = "meta-llama/Meta-Llama-3-70B-Instruct"
    test: Optional[bool] = None

# ----------------- API ----------------- #


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ----------------- Main ----------------- #


@app.post("/start_bot")
async def start_bot(request: Request) -> JSONResponse:
    runner_settings = RunnerSettings()
    try:
        request_body = await request.body()
        if len(request_body) > 0:
            runner_settings = RunnerSettings.model_validate_json(request_body)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {e}")
    except Exception as e:
        # If no data in request, pass
        pass

    # Is this a webhook creation request?
    if runner_settings.test is not None:
        return JSONResponse({"test": True})

    # Use specified room URL, or create a new one if not specified
    room_url = os.getenv("DAILY_SAMPLE_ROOM_URL", "")

    if not room_url:
        params = DailyRoomParams(
            properties=DailyRoomProperties()
        )
        try:
            room: DailyRoomObject = daily_rest_helper.create_room(params=params)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unable to provision room {e}")
    else:
        # Check passed room URL exists, we should assume that it already has a sip set up
        try:
            room: DailyRoomObject = daily_rest_helper.get_room_from_url(room_url)
        except Exception:
            raise HTTPException(
                status_code=500, detail=f"Room not found: {room_url}")

    # Give the agent a token to join the session
    token = daily_rest_helper.get_token(room.url, MAX_SESSION_TIME)

    if not room or not token:
        raise HTTPException(
            status_code=500, detail=f"Failed to get token for room: {room_url}")

    # Spawn a new agent, and join the user session
    # Note: this is mostly for demonstration purposes (refer to 'deployment' in README)
    try:
        bot_settings = BotSettings(
            room_url=room.url,
            room_token=token,
            prompt=runner_settings.prompt,
            deepgram_voice=runner_settings.deepgram_voice,
            deepgram_api_key=os.getenv("DEEPGRAM_API_KEY"),
            deepgram_base_url="http://0.0.0.0:8080/v1/speak",
            openai_model=runner_settings.openai_model,
            openai_api_key="ollama",
            openai_base_url="http://0.0.0.0:8000/v1",
        )
        bot_settings_str = bot_settings.model_dump_json(exclude_none=True)

        subprocess.Popen(
            [f"python3 -m bot -s '{bot_settings_str}'"],
            shell=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start subprocess: {e}")

    # Grab a token for the user to join with
    user_token = daily_rest_helper.get_token(room.url, MAX_SESSION_TIME)

    return JSONResponse({
        "room_url": room.url,
        "token": user_token,
    })


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
