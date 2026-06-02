#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
# Outbound WhatsApp calling example.
#
# Runs a FastAPI server that:
#   - Handles inbound WhatsApp call webhooks  (/whatsapp POST)
#   - Verifies the webhook during setup       (/whatsapp GET)
#   - Accepts a trigger to place outbound calls (/call-out POST)
#
# Usage:
#   1. Start ngrok:   ngrok http 7860
#   2. Copy .env.example -> .env and fill in credentials
#   3. Run:           uv run python examples/transports/transports-whatsapp.py
#   4. Place call:    curl -X POST http://localhost:7860/call-out \
#                          -H "Content-Type: application/json" \
#                          -d '{"to": "15551234567"}'
#
# Required env vars (see .env.example in same directory):
#   WHATSAPP_TOKEN, WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_APP_SECRET,
#   WHATSAPP_WEBHOOK_VERIFICATION_TOKEN,
#   OPENAI_API_KEY, DEEPGRAM_API_KEY, CARTESIA_API_KEY

import asyncio
import os
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.smallest.tts import SmallestTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.whatsapp.api import WhatsAppWebhookRequest
from pipecat.transports.whatsapp.client import WhatsAppClient

load_dotenv(override=True)

# Global client and HTTP session — initialised in the lifespan handler.
whatsapp_client: WhatsAppClient | None = None
_http_session: aiohttp.ClientSession | None = None


# ---------------------------------------------------------------------------
# Bot pipeline
# ---------------------------------------------------------------------------


async def run_bot(webrtc_connection: SmallWebRTCConnection) -> None:
    """Run a voice AI pipeline over an established WhatsApp WebRTC connection."""
    logger.info("Starting bot pipeline")

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = SmallestTTSService(
        api_key=os.environ["SMALLEST_API_KEY"],
        settings=SmallestTTSService.Settings(voice="sophia"),
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=(
                "You are a helpful voice assistant. Keep responses short and conversational."
            ),
        ),
    )

    context = LLMContext()
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_agg,
            llm,
            tts,
            transport.output(),
            assistant_agg,
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        logger.info("Client connected — starting conversation")
        context.add_message({"role": "developer", "content": "Greet the user warmly."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    await PipelineRunner(handle_sigint=False).run(task)


# ---------------------------------------------------------------------------
# App lifespan — initialise / tear down WhatsApp client
# ---------------------------------------------------------------------------


async def _on_call_status(call_id: str, status: str) -> None:
    logger.info(f"Call {call_id} status update: {status}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global whatsapp_client, _http_session

    _http_session = aiohttp.ClientSession()
    whatsapp_client = WhatsAppClient(
        whatsapp_token=os.environ["WHATSAPP_TOKEN"],
        phone_number_id=os.environ["WHATSAPP_PHONE_NUMBER_ID"],
        session=_http_session,
        whatsapp_secret=os.environ["WHATSAPP_APP_SECRET"],
        call_status_callback=_on_call_status,
    )
    logger.info("WhatsApp client initialised")

    yield

    logger.info("Shutting down — terminating all calls")
    await whatsapp_client.terminate_all_calls()
    await _http_session.close()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Webhook routes
# ---------------------------------------------------------------------------


@app.get("/whatsapp", summary="Verify WhatsApp webhook")
async def verify_webhook(request: Request):
    """Called by Meta once during webhook setup to confirm the URL is yours."""
    try:
        challenge = await whatsapp_client.handle_verify_webhook_request(
            params=dict(request.query_params),
            expected_verification_token=os.environ["WHATSAPP_WEBHOOK_VERIFICATION_TOKEN"],
        )
        return PlainTextResponse(str(challenge))
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/whatsapp", summary="Handle WhatsApp call events")
async def whatsapp_webhook(
    body: WhatsAppWebhookRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    x_hub_signature_256: str = Header(None),
):
    """Receives connect / status / terminate webhooks from WhatsApp."""
    if body.object != "whatsapp_business_account":
        raise HTTPException(status_code=400, detail="Unexpected object type")

    raw_body = await request.body()

    # Inbound calls: callback runs within this request's BackgroundTasks.
    async def inbound_connection_callback(connection: SmallWebRTCConnection):
        background_tasks.add_task(run_bot, connection)

    try:
        await whatsapp_client.handle_webhook_request(
            body,
            connection_callback=inbound_connection_callback,
            raw_body=raw_body,
            sha256_signature=x_hub_signature_256,
        )
        return {"status": "ok"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Webhook processing error: {e}")
        raise HTTPException(status_code=500, detail="Internal error")


# ---------------------------------------------------------------------------
# Outbound call trigger
# ---------------------------------------------------------------------------


@app.post("/call-out", summary="Place an outbound WhatsApp call")
async def call_out(request: Request):
    """
    Trigger an outbound call to a WhatsApp user.

    The business must have permission to call the destination number (granted
    when that user made an inbound call within the past 7 days, or via the
    call_permission_request API).

    Request body::

        {"to": "15551234567"}

    Returns::

        {"status": "calling", "call_id": "<id>"}
    """
    data = await request.json()
    to = data.get("to")
    if not to:
        raise HTTPException(status_code=400, detail="Missing 'to' field")

    # Outbound callback fires when the webhook returns the SDP answer.
    # asyncio.create_task is used because this callback will be invoked
    # from inside a *different* HTTP request (the /whatsapp webhook), so
    # the current request's BackgroundTasks scope is no longer active.
    async def outbound_connection_callback(connection: SmallWebRTCConnection):
        asyncio.create_task(run_bot(connection))

    try:
        call_id = await whatsapp_client.initiate_call(
            to=to,
            connection_callback=outbound_connection_callback,
        )
        logger.info(f"Outbound call initiated: call_id={call_id}, to={to}")
        return {"status": "calling", "call_id": call_id}
    except Exception as e:
        logger.error(f"Failed to initiate outbound call to {to}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
