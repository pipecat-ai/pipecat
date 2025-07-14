import json
import os
import shlex
import subprocess
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from fastapi.responses import JSONResponse
from utils.daily_helpers import create_daily_room

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.session = aiohttp.ClientSession()
    yield
    await app.state.session.close()


app = FastAPI(lifespan=lifespan)

# Add CORS middleware - ADD THIS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.post("/api/connect")
async def api_connect(request: Request):
    """Handle voice-ui-kit console connections."""
    print("Received voice-ui-kit console connection request")

    try:
        # For console connections, we can use default settings
        # or extract parameters from the request if needed
        data = (
            await request.json()
            if request.headers.get("content-type") == "application/json"
            else {}
        )

        # Use default phone number for console testing, or get from request
        phone_number = data.get("phone_number", "+1234567890")  # Default for testing
        caller_id = data.get("caller_id")

        # Create a Daily room
        room_details = await create_daily_room(request.app.state.session, phone_number)
        room_url = room_details["room_url"]
        token = room_details["token"]

        print(f"Created Daily room for console: {room_url}")

        # Prepare bot configuration for console connection
        bot_config = {"dialout_settings": {"phone_number": phone_number}}

        if caller_id:
            bot_config["dialout_settings"]["caller_id"] = caller_id

        body_json = json.dumps(bot_config)
        bot_cmd = f"python3 -m bot -u {room_url} -t {token} -b {shlex.quote(body_json)}"

        try:
            # Start the bot process
            subprocess.Popen(
                bot_cmd,
                shell=True,
                # Keep output visible for debugging
            )
            print(f"Started bot process for console: {bot_cmd}")
        except Exception as e:
            print(f"Error starting bot for console: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start bot: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in console connect: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

    # Return the format expected by voice-ui-kit
    return JSONResponse({"room_url": room_url, "token": token})


@app.post("/start")
async def handle_incoming_daily_webhook(request: Request) -> JSONResponse:
    """Handle dial-out request."""
    print("Received webhook from Daily")

    # Get the dial-in properties from the request
    try:
        data = await request.json()
        if "test" in data:
            # Pass through any webhook checks
            return JSONResponse({"test": True})

        if not data["dialout_settings"]:
            raise HTTPException(
                status_code=400, detail="Missing 'dialout_settings' in the request body"
            )

        if not data["dialout_settings"].get("phone_number"):
            raise HTTPException(
                status_code=400, detail="Missing 'phone_number' in dialout_settings"
            )

        # Extract the phone number we want to dial out to
        caller_phone = str(data["dialout_settings"]["phone_number"])
        print(f"Processing call to {caller_phone}")

        # Create a Daily room with dial-in capabilities
        try:
            room_details = await create_daily_room(request.app.state.session, caller_phone)
        except Exception as e:
            print(f"Error creating Daily room: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create Daily room: {str(e)}")

        room_url = room_details["room_url"]
        token = room_details["token"]
        print(f"Created Daily room: {room_url} with token: {token}")

        body_json = json.dumps(data)

        bot_cmd = f"python3 -m bot -u {room_url} -t {token} -b {shlex.quote(body_json)}"

        try:
            # CHANGE: Keep stdout/stderr for debugging
            # Start the bot in the background but capture output
            subprocess.Popen(
                bot_cmd,
                shell=True,
                # Don't redirect output so we can see logs
                # stdout=subprocess.DEVNULL,
                # stderr=subprocess.DEVNULL
            )
            print(f"Started bot process with command: {bot_cmd}")
        except Exception as e:
            print(f"Error starting bot: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start bot: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

    # Grab a token for the user to join with
    return JSONResponse({"room_url": room_url, "token": token})


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}


# ----------------- Main ----------------- #


if __name__ == "__main__":
    # Run the server
    port = int(os.getenv("PORT", "7860"))
    print(f"Starting server on port {port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
