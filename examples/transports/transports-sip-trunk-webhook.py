#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Receive Daily SIP-trunk notifications and start a voice agent per call.

A ``sip_client`` SIP trunk on Daily provisions a SIP account for each inbound
call and POSTs a signed ``sip_trunk.incoming`` notification to the trunk's
``notification.webhook_url``; the caller hears ringback until that endpoint
answers 2xx and a bot registers on the account. This server is such an
endpoint. It takes the steps Pipecat Cloud's handler takes — the probe, the
field checks, the signature over the request bytes — and starts the dial-in
voice agent of ``transports-sip-dialin.py`` on the notified account. Pipecat
Cloud reads the trunk's secret over an internal Daily API; here it comes from
``SIP_TRUNK_HMAC``.

Run it where Daily can reach it, through a tunnel in development::

    SIP_TRUNK_HMAC=... uv run python transports-sip-trunk-webhook.py --port 7860
    ngrok http 7860

then create the trunk against it (the create fails unless the server answers
Daily's ``sip_trunk.test`` probe)::

    curl -s -X POST "$DAILY_API_URL/sip-trunk" \\
      -H "Authorization: Bearer $DAILY_API_KEY" -H "Content-Type: application/json" \\
      -d '{"trunk_name": "mytrunk", "trunk_type": "sip_client",
           "trunk_config": {"is_open": true, "exp_offset": 3600},
           "sip_client_template": {"transport": "tls"},
           "notification": {"webhook_url": "https://<host>/daily-sip-trunk-webhook"}}'

The response carries ``config.notification.hmac``, the value of
``SIP_TRUNK_HMAC``, and ``sip_uri``, the host your carrier's trunk originates
calls to. The secret does not exist before the create returns, so for the
first create start the server with any value and restart it with the real one;
the probe is answered without verification either way. Restart it the same
way after a secret rotation.

Environment: ``SIP_TRUNK_HMAC`` and the agent's ``DEEPGRAM_API_KEY``,
``OPENAI_API_KEY`` and ``CARTESIA_API_KEY``. The ``SIP_*`` settings of
:mod:`pipecat.runner.sip` (``SIP_TRANSPORT``, ``SIP_STUN_SERVER``, the jitter
buffer, tracing) apply to the account each notification carries; a
notification that names its own transport wins.

What an answer means to the caller: a 2xx is a commitment, after which Daily
holds the caller until the bot's REGISTER lands at the registrar; anything
else is retried inside Daily's budget (about 12 seconds from the INVITE) and
the call is then rejected with ``480``.
"""

import argparse
import asyncio
import base64
import hashlib
import hmac
import json
import os
import sys

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.sip import configure, sip_runner_arguments
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.sip.transport import SIPParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

# The trunk's notification.hmac, as POST /sip-trunk returned it.
SIP_TRUNK_HMAC = os.getenv("SIP_TRUNK_HMAC", "")

# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "sip": lambda: SIPParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="86e30c1d-714b-4074-a1f2-1cb6b552fb49",
        ),
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction="You are a helpful assistant on a phone call. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way.",
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation.
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_dialin_ready")
    async def on_dialin_ready(transport, sip_endpoint):
        # REGISTER OK — the account is routable. Never fires in trunk mode
        # (reg_interval=0); use on_connected as the ready signal there.
        logger.info(f"Ready for dial-in: {sip_endpoint}")

    @transport.event_handler("on_dialin_connected")
    async def on_dialin_connected(transport, data):
        logger.info(f"Dial-in connected: {data}")

    @transport.event_handler("on_dialin_stopped")
    async def on_dialin_stopped(transport, data):
        logger.info(f"Dial-in stopped: {data}")

    @transport.event_handler("on_dialin_error")
    async def on_dialin_error(transport, data):
        logger.error(f"Dial-in error: {data.get('errorMsg')}")

    @transport.event_handler("on_dialin_warning")
    async def on_dialin_warning(transport, data):
        # Non-fatal media-health warnings (e.g. sustained packet loss or a
        # starved transmit buffer) — the call continues.
        logger.warning(f"Dial-in warning: {data.get('errorMsg')}")

    @transport.event_handler("on_dtmf_event")
    async def on_dtmf_event(transport, data):
        # Keypad presses also flow into the pipeline as InputDTMFFrame.
        logger.info(f"DTMF received: {data['tone']}")

    @transport.event_handler("on_call_quality_stats")
    async def on_call_quality_stats(transport, stats):
        # SIP-only: the call's final quality snapshot (duration, packet
        # loss, jitter, RTT, ...) at call close.
        logger.info(f"Call quality: {stats}")

    @transport.event_handler("on_error")
    async def on_error(transport, error):
        # Fatal stack or registration failures (bad credentials,
        # unreachable server).
        logger.error(f"Transport error: {error}")
        await runner.cancel()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


def verify_signature(timestamp: str, signature: str, raw_body: bytes, hmac_b64: str) -> bool:
    """Check a notification's X-Siptrunk-Signature.

    Daily signs each notification with the trunk's ``notification.hmac``: the
    header is the base64 HMAC-SHA256, keyed with the base64-decoded secret, of
    ``"<X-Siptrunk-Timestamp>.<request body>"``. The body must be the bytes as
    received; a re-serialized body only matches while both ends happen to
    serialize alike.
    """
    if not timestamp or not signature or not hmac_b64:
        return False
    try:
        key = base64.b64decode(hmac_b64)
    except ValueError:
        return False
    digest = hmac.new(key, f"{timestamp}.".encode() + raw_body, hashlib.sha256).digest()
    return hmac.compare_digest(base64.b64encode(digest), signature.encode())


app = FastAPI()

# Bots started by notifications, held so their tasks are not garbage-collected.
_sessions: set[asyncio.Task] = set()


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


@app.post("/daily-sip-trunk-webhook")
async def daily_sip_trunk_webhook(request: Request):
    """Verify a SIP-trunk notification and start a bot on its account.

    The steps are Pipecat Cloud's handler's, in its order; only where the
    secret comes from differs.
    """

    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type != "application/json":
        raise _bad_request("Unsupported body: expected application/json")
    raw = await request.body()
    try:
        data = json.loads(raw)
    except ValueError:
        raise _bad_request("Invalid JSON payload")
    if not isinstance(data, dict):
        raise _bad_request("Invalid JSON payload")

    # The config-time probe is signed with the trunk's new secret before the
    # create call has returned it, so it cannot be verified; the 2xx is what
    # lets the create or update succeed.
    if data.get("event") == "sip_trunk.test":
        logger.debug("SIP-trunk webhook probe received")
        return {"status": "OK"}

    if data.get("event") != "sip_trunk.incoming":
        raise _bad_request("Unexpected SIP-trunk event: expected sip_trunk.incoming")
    sip_client = data.get("sip_client")
    if not sip_client and data.get("room"):
        raise _bad_request(
            "This webhook starts a bot for a sip_client trunk; a room trunk is not supported"
        )
    trunk = data.get("trunk")
    call = data.get("call")
    if not (
        isinstance(trunk, dict)
        and trunk.get("id")
        and isinstance(call, dict)
        and call.get("sipCallIdHeader")
        and isinstance(sip_client, dict)
        and all(sip_client.get(key) for key in ("username", "password", "domain"))
    ):
        raise _bad_request(
            "Missing required SIP-trunk fields: trunk.id, call.sipCallIdHeader, "
            "sip_client.username, sip_client.password, or sip_client.domain"
        )

    signature = request.headers.get("x-siptrunk-signature")
    timestamp = request.headers.get("x-siptrunk-timestamp")
    if not signature or not timestamp:
        raise HTTPException(status_code=401, detail="Missing webhook signature headers")
    # The signature covers the bytes Daily sent, not a re-serialization of the
    # parsed body.
    call_id = call["sipCallIdHeader"]
    if not verify_signature(timestamp, signature, raw, SIP_TRUNK_HMAC):
        logger.warning(f"SIP-trunk call {call_id}: invalid notification signature")
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    # No dedupe here, as on Pipecat Cloud. Daily re-delivers a notification whose
    # earlier attempt failed, and marks a carrier failover between its SIP nodes
    # with call.duplicate: a second branch with the same Call-ID and its own
    # account. Both must be answered 2xx and served, since one of the two
    # branches carries the caller and it is often the flagged one. A handler
    # that keeps state keys it on (call.sipCallIdHeader, sip_client.username).
    # See Daily's reference for the sip_trunk.incoming notification.
    logger.info(
        f"SIP-trunk call {call_id} from {call.get('From')} to {call.get('To')}"
        f"{' (duplicate)' if call.get('duplicate') else ''}: starting the bot"
    )

    async with aiohttp.ClientSession() as session:
        config = await configure(session, body=data)
    base = RunnerArguments(body=data, session_id=call_id)
    try:
        runner_args = sip_runner_arguments(config, base)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Answer now: the registrar holds the caller until the bot's REGISTER lands.
    task = asyncio.create_task(bot(runner_args))
    _sessions.add(task)
    task.add_done_callback(_sessions.discard)
    return {"status": "OK", "sessionId": call_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily SIP-trunk notification webhook")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9191, help="Port to listen on")
    args = parser.parse_args()

    if not SIP_TRUNK_HMAC:
        sys.exit(
            "SIP_TRUNK_HMAC is required: the trunk's notification.hmac, as POST /sip-trunk "
            "returned it."
        )
    uvicorn.run(app, host=args.host, port=args.port)
