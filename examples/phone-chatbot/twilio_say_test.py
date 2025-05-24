import os
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
import logging
import uvicorn

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger("twilio_say_test")

# Load environment variables from .env
load_dotenv(override=True)

# Twilio REST client
from twilio.rest import Client as TwilioClient

app = FastAPI()
TWILIO = TwilioClient(
    os.getenv("TWILIO_ACCOUNT_SID"),
    os.getenv("TWILIO_AUTH_TOKEN"),
)
CALLER_ID = os.getenv("TWILIO_CALLER_ID")
PUBLIC_HOST = os.getenv("PUBLIC_HOSTNAME")

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
        url=f"https://{PUBLIC_HOST}/twiml_say",
        record=True,
    )
    
    logger.info("Twilio call SID=%s", call.sid)
    return {"status": "calling", "call_sid": call.sid}

@app.api_route("/twiml_say", methods=["GET", "POST"])
async def twiml_say(request: Request):
    logger.info("Twilio requested TwiML for Say test")
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice" loop="3">
    This is a test call using Twilio's built-in text to speech. 
    If you can hear this, then the issue is with your streaming implementation, not with Twilio.
  </Say>
  <Pause length="2"/>
  <Say voice="alice">Repeating one more time. This is a test.</Say>
</Response>"""
    return PlainTextResponse(content=twiml, media_type="application/xml")

# TwiML Bin URL: https://handler.twilio.com/twiml/EH6cbd8edefc48d70a8cce1c17de4c97db
# Contains:
# <?xml version="1.0" encoding="UTF-8"?>
# <Response>
#   <Say voice="alice" loop="3">This is a test call using Twilio's built-in text to speech. 
#   If you can hear this, then the issue is with your streaming implementation, not with Twilio.</Say>
#   <Pause length="2"/>
#   <Say voice="alice">Repeating one more time. This is a test.</Say>
# </Response>

if __name__ == "__main__":
    uvicorn.run(
        "twilio_say_test:app",
        host="0.0.0.0", 
        port=8000,
        log_level="debug"
    ) 