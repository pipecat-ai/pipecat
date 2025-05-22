#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""server.py.

Webhook server to handle webhook coming from Daily, create a Daily room and start the bot.
"""

import json
import os
import shlex
import subprocess
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from utils.daily_helpers import create_daily_room

load_dotenv()

# ----------------- API ----------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create aiohttp session to be used for Daily API calls
    app.state.session = aiohttp.ClientSession()
    yield
    # Close session when shutting down
    await app.state.session.close()


app = FastAPI(lifespan=lifespan)


def extract_phone_from_sip_uri(sip_uri):
    """Extract phone number from SIP URI.

    Args:
        sip_uri: SIP URI in format "sip:+17868748498@daily-twilio-integration.sip.twilio.com"

    Returns:
        Phone number string (e.g., "+17868748498") or None if invalid format
    """
    if not sip_uri or not isinstance(sip_uri, str):
        return None

    if sip_uri.startswith("sip:") and "@" in sip_uri:
        phone_part = sip_uri[4:]  # Remove 'sip:' prefix
        caller_phone = phone_part.split("@")[0]  # Get everything before '@'
        return caller_phone
    return None


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

        if not data["dialout_settings"].get("sip_uri"):
            raise HTTPException(status_code=400, detail="Missing 'sip_uri' in dialout_settings")

        # Extract the phone number we want to dial out to
        sip_uri = str(data["dialout_settings"]["sip_uri"])
        caller_phone = extract_phone_from_sip_uri(sip_uri)
        print(f"SIP URI: {sip_uri}")
        print(f"Processing sip call to {caller_phone}")

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
