import argparse
import json
import os

import httpx
import uvicorn
from bot import run_bot
from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from jinja2 import Environment, FileSystemLoader
from loguru import logger
from starlette.responses import HTMLResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/")
async def start_call():
    print("POST TwiML")
    return HTMLResponse(content=open("templates/streams.xml").read(), media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    start_data = websocket.iter_text()
    await start_data.__anext__()
    call_data = json.loads(await start_data.__anext__())
    print(call_data, flush=True)
    stream_sid = call_data["start"]["streamSid"]
    call_sid = call_data["start"]["callSid"]
    print("WebSocket connection accepted")
    await run_bot(websocket, stream_sid, call_sid, app.state.testing)


DEFAULT_TEMPLATE_ENVIRONMENT = Environment(
    loader=FileSystemLoader("%s/templates/" % os.path.dirname(__file__))
)


def render_template(template_name: str, template_environment: Environment, **kwargs):
    template = template_environment.get_template(template_name)
    return template.render(**kwargs)


def get_connection_twiml(environment: Environment):
    return Response(
        render_template(
            template_name="streams.xml",
            template_environment=environment,
        ),
        media_type="application/xml",
    )


@app.post("/twilio/outbound")
async def outbound_call(request: Request):
    body = await request.json()
    to = body["to"]
    data = {
        "Twiml": get_connection_twiml(DEFAULT_TEMPLATE_ENVIRONMENT).body.decode("utf-8"),
        "To": f"{to}",
        "From": f"+18287528082",
        "Record": True,
        "RecordingStatusCallback": f"https://{os.getenv('SERVER_BASE_URL')}/webhook/twilio/recording_status",
    }
    basic_auth = httpx.BasicAuth(
        username=os.getenv("TWILIO_ACCOUNT_SID"),
        password=os.getenv("TWILIO_AUTH_TOKEN"),
    )
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{os.getenv('TWILIO_ACCOUNT_SID')}/Calls.json",
            auth=basic_auth,
            data=data,
        )

        return response.json()["sid"]

        # if not response.ok:
        #     if response.status == 400:
        #         print(await response.json())
        # response = await response.json()
        # return response["sid"]


@app.post("/webhook/twilio/recording_status")
async def handle_twilio_recording_status_callback(request: Request):
    """This function will handle the twilio recording status callback."""
    try:
        form = await request.form()
        call_data = dict(form)

        call_sid = call_data["CallSid"]
        recording_url = call_data["RecordingUrl"]

        logger.info(f"call_sid: {call_sid}, recording_url: {recording_url}")

    except Exception as e:
        logger.error(f"Failed to handle twilio recording status callback: {str(e)}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Twilio Chatbot Server")
    parser.add_argument(
        "-t", "--test", action="store_true", default=False, help="set the server in testing mode"
    )
    args, _ = parser.parse_known_args()

    app.state.testing = args.test

    uvicorn.run(app, host="0.0.0.0", port=8765)
