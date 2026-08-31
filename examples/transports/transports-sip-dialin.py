#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice agent on the SIP transport.

The bot registers with a SIP server and answers incoming calls. Run it with
an account on any SIP server::

    SIP_USER=1001 SIP_PASS=secret SIP_DOMAIN=sip.example.com \\
        python transports-sip-dialin.py -t sip

The same bot runs unchanged under ``-t daily``, ``-t webrtc``, or ``-t eval``
— only the transport params entry differs.

Under ``-t daily``, SIP dial-in needs the Daily room created with the ``sip``
property (a plain run creates a standard browser room). Ask the dev runner
for one through its ``/start`` endpoint::

    python transports-sip-dialin.py -t daily
    curl -X POST http://localhost:7860/start \\
        -H "Content-Type: application/json" \\
        -d '{"createDailyRoom": true, "dailyRoomProperties": {"sip": {}}}'

then dial the SIP URI that ``on_dialin_ready`` logs, from any SIP client.
For a PSTN number, point the number's pinless dial-in webhook at the
runner's ``/daily-dialin-webhook`` endpoint instead — the runner creates a
SIP-enabled room per call and the transport forwards the caller into it.

``SIP_AUDIO_CODECS`` (comma-separated, e.g. ``opus/48000/2,PCMU/8000/1``)
sets the codec preference order. For knobs beyond the ``SIP_*`` environment
variables, construct ``SIPConnection(...)`` yourself and pass it to
``SIPTransport`` instead of using ``create_transport``::

    from pipecat.transports.sip.connection import SIPConnection
    from pipecat.transports.sip.transport import SIPParams, SIPTransport

    connection = SIPConnection(
        user="1001",
        domain="sip.example.com",
        password="secret",
        reg_interval=0,  # trunk mode: no registration — peer must reach this host directly with static IP address (no AOR)
        auth_user="trunk-user",  # digest username, when it differs from user
        dtmf_mode="info",  # DTMF as SIP INFO instead of RFC 4733
        net_interface="10.0.0.5",  # force a specific egress interface
        audio_codecs=("opus/48000/2", "PCMU/8000/1"),
    )
    transport = SIPTransport(
        connection, SIPParams(audio_in_enabled=True, audio_out_enabled=True)
    )
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
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
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.sip.transport import SIPParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "sip": lambda: SIPParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
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


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
