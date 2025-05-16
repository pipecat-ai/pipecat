"""Webhook server to handle Twilio calls and start the voice bot."""

import os
import shlex
import subprocess
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse
from utils.daily_helpers import create_sip_room

# Load environment variables
load_dotenv()


# Initialize FastAPI app with aiohttp session
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create aiohttp session to be used for Daily API calls
    app.state.session = aiohttp.ClientSession()
    yield
    # Close session when shutting down
    await app.state.session.close()


app = FastAPI(lifespan=lifespan)


@app.post("/call", response_class=PlainTextResponse)
async def handle_call(request: Request):
    """Handle incoming Twilio call webhook."""
    print("Received call webhook from Twilio")

    try:
        # Get form data from Twilio webhook
        form_data = await request.form()
        data = dict(form_data)

        # Extract call ID (required to forward the call later)
        call_sid = data.get("CallSid")
        if not call_sid:
            raise HTTPException(status_code=400, detail="Missing CallSid in request")

        # Extract the caller's phone number
        caller_phone = str(data.get("From", "unknown-caller"))
        print(f"Processing call with ID: {call_sid} from {caller_phone}")

        # Create a Daily room with SIP capabilities
        try:
            room_details = await create_sip_room(request.app.state.session, caller_phone)
        except Exception as e:
            print(f"Error creating Daily room: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create Daily room: {str(e)}")

        # Extract necessary details
        room_url = room_details["room_url"]
        token = room_details["token"]
        sip_endpoint = room_details["sip_endpoint"]

        # Make sure we have a SIP endpoint
        if not sip_endpoint:
            raise HTTPException(status_code=500, detail="No SIP endpoint provided by Daily")

        # Start the bot process
        bot_cmd = f"python bot.py -u {room_url} -t {token} -i {call_sid} -s {sip_endpoint}"
        try:
            # Use shlex to properly split the command for subprocess
            cmd_parts = shlex.split(bot_cmd)

            # CHANGE: Keep stdout/stderr for debugging
            # Start the bot in the background but capture output
            subprocess.Popen(
                cmd_parts,
                # Don't redirect output so we can see logs
                # stdout=subprocess.DEVNULL,
                # stderr=subprocess.DEVNULL
            )
            print(f"Started bot process with command: {bot_cmd}")
        except Exception as e:
            print(f"Error starting bot: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start bot: {str(e)}")

        # Generate TwiML response to put the caller on hold with music
        # You can replace the URL with your own music file
        # or use Twilio's built-in music on hold
        # https://www.twilio.com/docs/voice/twiml/play#music-on-hold
        resp = VoiceResponse()
        resp.play(
            url="https://therapeutic-crayon-2467.twil.io/assets/US_ringback_tone.mp3",
            loop=10,
        )

        return str(resp)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    # Run the server
    port = int(os.getenv("PORT", "8000"))
    print(f"Starting server on port {port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
