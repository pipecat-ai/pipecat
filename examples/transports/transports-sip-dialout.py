#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice agent that places a SIP dial-out call.

The bot connects to the SIP server, dials a destination, and starts the
conversation when the callee answers. Give it a destination with an
environment variable::

    SIP_USER=1001 SIP_PASS=secret SIP_DOMAIN=sip.example.com \\
        SIP_DIALOUT_URI=sip:2002@sip.example.com \\
        python transports-sip-dialout.py -t sip

or with a runner body file (``--runner-body body.json``)::

    {"dialout_settings": {"sipUri": "sip:2002@sip.example.com"}}

To dial a phone number instead, set ``SIPParams.trunk`` and use
``{"phoneNumber": "+15551234567"}`` as the dial-out settings — the call
INVITEs ``sip:+15551234567@<trunk>``. The caller ID shown to the callee is
the connection's ``user`` (it becomes the From header); there is no per-call
caller ID as on Daily — presenting a different one means a different
``SIPConnection``.

For Twilio Elastic SIP Trunking, the caller ID must be a number on your
trunk (or a verified caller ID), authentication uses the trunk's credential
list, and ``reg_interval=0`` is required — Twilio termination does not
accept REGISTER::

    SIP_USER=+15559876543 \\
        SIP_DOMAIN=my-trunk.pstn.twilio.com \\
        SIP_AUTH_USER=my-cred-user SIP_PASS=my-cred-pass \\
        SIP_REG_INTERVAL=0 \\
        SIP_DIALOUT_URI=sip:+15551234567@my-trunk.pstn.twilio.com \\
        python transports-sip-dialout.py -t sip

Set ``SIP_DIALOUT_DTMF`` (e.g. ``1234#``) to send DTMF tones once the
callee answers — useful for navigating an IVR.

The same bot dials out through Daily under ``-t daily``: the Daily room must
be created with ``enable_dialout`` (a plain run creates a room without it),
which requires the dial-out entitlement on your Daily domain. Ask the dev
runner for one through its ``/start`` endpoint::

    SIP_DIALOUT_URI=sip:2002@sip.example.com \\
        python transports-sip-dialout.py -t daily
    curl -X POST http://localhost:7860/start \\
        -H "Content-Type: application/json" \\
        -d '{"createDailyRoom": true, "dailyRoomProperties": {"enable_dialout": true}}'

The bot dials from ``on_connected`` (Daily fires it right after joining) and
the same ``on_dialout_*`` handlers cover both transports.
"""

import os

from dotenv import load_dotenv
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
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.sip.transport import SIPParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

transport_params = {
    "sip": lambda: SIPParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        # This bot places calls; leave incoming INVITEs unanswered.
        auto_answer=False,
    ),
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


def get_dialout_settings(runner_args: RunnerArguments) -> dict | None:
    """The dial-out settings from the runner body, or SIP_DIALOUT_URI."""
    body = runner_args.body or {}
    if isinstance(body, dict) and "dialout_settings" in body:
        return body["dialout_settings"]
    uri = os.getenv("SIP_DIALOUT_URI")
    return {"sipUri": uri} if uri else None


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments, dialout_settings: dict):
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
            system_instruction="You are a helpful assistant on a phone call you placed. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way.",
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

    @transport.event_handler("on_connected")
    async def on_connected(transport, data=None):
        logger.info(f"Dialing {dialout_settings}")
        session_id, error = await transport.start_dialout(dialout_settings)
        if error:
            logger.error(f"Dial-out failed: {error}")
            await runner.cancel()

    @transport.event_handler("on_dialout_connected")
    async def on_dialout_connected(transport, data):
        logger.info(f"Ringing: {data}")

    @transport.event_handler("on_dialout_answered")
    async def on_dialout_answered(transport, data):
        logger.info(f"Answered: {data}")
        tones = os.getenv("SIP_DIALOUT_DTMF")
        if tones:
            error = await transport.send_dtmf({"tones": tones})
            if error:
                logger.error(f"send_dtmf failed: {error}")

    @transport.event_handler("on_dialout_stopped")
    async def on_dialout_stopped(transport, data):
        logger.info(f"Dial-out stopped: {data}")

    @transport.event_handler("on_dialout_error")
    async def on_dialout_error(transport, data):
        logger.error(f"Dial-out error: {data.get('errorMsg')}")
        await runner.cancel()

    @transport.event_handler("on_dialout_warning")
    async def on_dialout_warning(transport, data):
        # Non-fatal media-health warnings (e.g. sustained packet loss or a
        # starved transmit buffer) — the call continues.
        logger.warning(f"Dial-out warning: {data.get('errorMsg')}")

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

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Kick off the conversation once the callee answers.
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    dialout_settings = get_dialout_settings(runner_args)
    if not dialout_settings:
        logger.error(
            "No dial-out destination: set SIP_DIALOUT_URI or pass --runner-body with "
            '{"dialout_settings": {"sipUri": ...}}.'
        )
        return

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args, dialout_settings)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
