"""Webhook server to handle Twilio calls and start the voice bot with silence detection and logging."""

import os
import shlex
import subprocess
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse, Gather
from utils.daily_helpers import create_sip_room

# Load environment variables
load_dotenv()

# Initialize FastAPI app with aiohttp session
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.session = aiohttp.ClientSession()
    app.state.unanswered = {}  # Track missed responses per call
    yield
    await app.state.session.close()


app = FastAPI(lifespan=lifespan)


@app.post("/call", response_class=PlainTextResponse)
async def handle_call(request: Request):
    """Handle incoming Twilio call webhook."""
    print("Received call webhook from Twilio")

    try:
        form_data = await request.form()
        data = dict(form_data)
        call_sid = data.get("CallSid")
        if not call_sid:
            raise HTTPException(status_code=400, detail="Missing CallSid in request")

        caller_phone = str(data.get("From", "unknown-caller"))
        print(f"Processing call with ID: {call_sid} from {caller_phone}")

        try:
            room_details = await create_sip_room(request.app.state.session, caller_phone)
        except Exception as e:
            print(f"Error creating Daily room: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create Daily room: {str(e)}")

        room_url = room_details["room_url"]
        token = room_details["token"]
        sip_endpoint = room_details["sip_endpoint"]

        if not sip_endpoint:
            raise HTTPException(status_code=500, detail="No SIP endpoint provided by Daily")

        bot_cmd = f"python bot.py -u {room_url} -t {token} -i {call_sid} -s {sip_endpoint}"
        try:
            subprocess.Popen(shlex.split(bot_cmd))
            print(f"Started bot process with command: {bot_cmd}")
        except Exception as e:
            print(f"Error starting bot: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start bot: {str(e)}")

        # Respond with a Gather to detect silence
        resp = VoiceResponse()
        gather = Gather(
            input="speech",
            action="/gather-response",
            timeout=10
        )
        gather.say("Hello! Please say something so we can assist you.")
        resp.append(gather)
        resp.say("Are you still there?")
        return str(resp)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.post("/gather-response", response_class=PlainTextResponse)
async def gather_response(request: Request):
    """Handle responses or silence after Gather."""
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    speech_result = form_data.get("SpeechResult", "").strip()

    missed = app.state.unanswered.get(call_sid, 0)

    if not speech_result:
        missed += 1
    else:
        missed = 0

    app.state.unanswered[call_sid] = missed

    resp = VoiceResponse()
    if missed >= 3:
        resp.say("We couldn't hear you. Ending the call. Goodbye.")
        resp.hangup()
    else:
        gather = Gather(
            input="speech",
            action="/gather-response",
            timeout=10
        )
        gather.say("Can you please respond?")
        resp.append(gather)
        resp.say("Still there?")
    return str(resp)


@app.post("/status-callback")
async def status_callback(request: Request):
    """Log post-call summary."""
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    duration = form_data.get("CallDuration", 0)

    missed = app.state.unanswered.get(call_sid, 0)
    print(f"\n📞 Call Summary:")
    print(f"- Call SID: {call_sid}")
    print(f"- Duration: {duration} seconds")
    print(f"- Missed Prompts: {missed}\n")

    return PlainTextResponse("OK")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"Starting server on port {port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
