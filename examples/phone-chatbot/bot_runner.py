import argparse
import os
from contextlib import asynccontextmanager

import aiohttp
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from runner_code import RoomManager, RoomRequest, RoomStateManager, WebhookHandler

from pipecat.transports.services.helpers.daily_rest import (
    DailyRESTHelper,
    DailyRoomObject,
    DailyRoomParams,
    DailyRoomProperties,
    DailyRoomSipParams,
)

# Load environment variables
load_dotenv(override=True)

# ------------ Configuration ------------ #
MAX_SESSION_TIME = 5 * 60  # 5 minutes
REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "DAILY_API_KEY", "ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"]

daily_helpers = {}
room_state_manager = None
webhook_handler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Setup and teardown of application resources."""
    # Set up aiohttp session and Daily helper
    aiohttp_session = aiohttp.ClientSession()
    daily_helpers["rest"] = DailyRESTHelper(
        daily_api_key=os.getenv("DAILY_API_KEY", ""),
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )

    # Initialize managers
    global room_state_manager, webhook_handler
    room_state_manager = RoomStateManager(
        daily_api_key=os.getenv("DAILY_API_KEY"),
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        daily_helpers=daily_helpers,
    )
    webhook_handler = WebhookHandler(room_state_manager)

    yield

    # Cleanup
    await aiohttp_session.close()


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and their responses."""
    print(f"Incoming request: {request.method} {request.url}")
    print(f"Headers: {request.headers}")
    body = await request.body()
    print(f"Body: {body}")
    response = await call_next(request)
    print(f"Response status: {response.status_code}")
    return response


@app.post("/daily_start_bot")
async def create_dialin_room(request: Request) -> JSONResponse:
    """
    Create a new Daily room based on the request parameters.

    The room can be created for different purposes:
    1. Voicemail detection only
    2. Dialout with voicemail detection
    3. User dial-in
    4. Operator room
    """
    try:
        data = await request.json()
        if "test" in data:
            return JSONResponse({"test": True})

        # Create room request and initialize room manager
        room_request = RoomRequest.from_json(data)
        room_manager = RoomManager(
            daily_helpers=daily_helpers, room_url=os.getenv("DAILY_SAMPLE_ROOM_URL")
        )

        # Create the room and get result
        result = await room_manager.create_room(room_request)
        room = result["room"]  # Get DailyRoomObject

        # Store room state for webhook handling
        room_state_manager.store_created_room(
            room=room,
            call_id=room_request.call_id,
            call_domain=room_request.call_domain,
            is_operator_room=result["is_operator_room"],
        )

        return JSONResponse({"room_url": room.url, "sipUri": room.config.sip_endpoint})

    except Exception as e:
        print(f"Error handling request: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process room creation request: {str(e)}"
        )


@app.post("/joined_room")
async def joined_room_webhook(request: Request) -> JSONResponse:
    """Handle webhook notifications when participants join a room."""
    return await webhook_handler.handle_join_webhook(request)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse({"status": "healthy"})


def check_environment():
    """Verify all required environment variables are set."""
    missing_vars = [var for var in REQUIRED_ENV_VARS if var not in os.environ]
    if missing_vars:
        raise Exception(f"Missing environment variables: {', '.join(missing_vars)}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument(
        "--host", type=str, default=os.getenv("HOST", "0.0.0.0"), help="Host address"
    )
    parser.add_argument("--port", type=int, default=os.getenv("PORT", 7860), help="Port number")
    parser.add_argument("--reload", action="store_true", default=True, help="Reload code on change")

    config = parser.parse_args()

    try:
        # Check environment variables
        check_environment()

        # Start the server
        import uvicorn

        uvicorn.run(
            "bot_runner:app",  # This stays the same because bot_runner.py is in the root
            host=config.host,
            port=config.port,
            reload=config.reload,
        )
    except KeyboardInterrupt:
        print("Pipecat runner shutting down...")
    except Exception as e:
        print(f"Failed to start server: {str(e)}")
        raise
