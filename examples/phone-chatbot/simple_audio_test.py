import os
import base64
import json
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
import asyncio
import uvicorn
import logging

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger("simple_audio_test")

# Load environment variables from .env
load_dotenv(override=True)

# Twilio REST client
from twilio.rest import Client as TwilioClient

app = FastAPI()
PUBLIC_HOST = os.getenv("PUBLIC_HOSTNAME")
TWILIO = TwilioClient(
    os.getenv("TWILIO_ACCOUNT_SID"),
    os.getenv("TWILIO_AUTH_TOKEN"),
)
CALLER_ID = os.getenv("TWILIO_CALLER_ID")

# Create a simple 8kHz mulaw audio message - just a beep tone
SIMPLE_AUDIO = bytes([
    # This is a simple 8kHz mulaw-encoded beep sound
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
]) * 1000  # Repeat to make it longer

@app.post("/start_call")
async def start_call(req: Request):
    data = await req.json()
    to_number = data.get("to")
    logger.info("Starting call to %s", to_number)
    
    if not to_number:
        return PlainTextResponse("Missing 'to' in JSON body", status_code=400)
    
    call = TWILIO.calls.create(
        to=to_number,
        from_=CALLER_ID,
        url=f"https://{PUBLIC_HOST}/simple_twiml", 
        record=True,
    )
    
    logger.info("Twilio call SID=%s", call.sid)
    return {"status": "calling", "call_sid": call.sid}

@app.api_route("/simple_twiml", methods=["GET", "POST"])
async def simple_twiml(request: Request):
    logger.info("Twilio requested TwiML for call")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{PUBLIC_HOST}/ws/simple_audio">
      <Parameter name="track" value="both" />
      <Parameter name="codec" value="pcmu" />
      <Parameter name="sample_rate" value="8000" />
    </Stream>
  </Connect>
  <Pause length="60"/>
</Response>"""
    return PlainTextResponse(content=twiml, media_type="application/xml")

@app.websocket("/ws/simple_audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    try:
        # Receive Twilio's initial 'connected' or 'start' message
        msg = await websocket.receive_text()
        data = json.loads(msg)
        logger.info("Received message: %s", data)
        
        # Wait a moment for connection to stabilize
        await asyncio.sleep(1)
        
        # Send a media message with our audio
        logger.info("Sending audio data (%d bytes)", len(SIMPLE_AUDIO))
        media_msg = {
            "event": "media", 
            "media": {
                "track": "outbound",
                "chunk": base64.b64encode(SIMPLE_AUDIO).decode('utf-8'),
                "timestamp": "0",
                "payload": "audio"
            }
        }
        await websocket.send_text(json.dumps(media_msg))
        logger.info("Audio sent!")
        
        # Keep the connection alive
        count = 0
        while True:
            count += 1
            if count % 100 == 0:
                logger.info("Still connected, sending audio again")
                await websocket.send_text(json.dumps(media_msg))
            
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
                logger.debug("Received message: %s", msg)
            except asyncio.TimeoutError:
                # No message received, continue
                pass
                
    except Exception as e:
        logger.exception("Error in WebSocket: %s", e)
    finally:
        logger.info("WebSocket connection closed")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(
        "simple_audio_test:app",
        host="0.0.0.0", 
        port=8000,
        log_level="debug"
    ) 