#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""FreeSWITCH SIP Transport Example (LAN / Dial-in Only)

Accepts incoming SIP calls routed by FreeSWITCH and runs a full
STT -> LLM -> TTS conversational agent pipeline per call. Audio is
exchanged over RTP using G.711 codecs with automatic 8kHz/16kHz resampling.

Constraints:
    - LAN-only: no NAT traversal, STUN, or ICE support.
    - Dial-in only: no SIP REGISTER; relies on FreeSWITCH to route calls.
    - UAS role only: the bot does not originate outbound calls.

Typical setup:
    1. Configure FreeSWITCH to route calls to this bot's IP:port
    2. Set environment variables (see below)
    3. Run this script
    4. Place a call through FreeSWITCH to reach the bot

Testing with pjsua (CLI SIP client):
    pjsua --null-audio sip:bot@<HOST>:5060

Environment variables:
    DEEPGRAM_API_KEY: Deepgram STT API key
    OPENAI_API_KEY:   OpenAI LLM API key
    CARTESIA_API_KEY: Cartesia TTS API key
    SIP_HOST:         Bind address (default: 0.0.0.0)
    SIP_PORT:         SIP port (default: 5060)
"""

import asyncio
import logging
import os

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
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.sip import (
    FreeSwitchSIPCallTransport,
    FreeSwitchSIPParams,
    FreeSwitchSIPServerTransport,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    sip_host = os.getenv("SIP_HOST", "0.0.0.0")
    sip_port = int(os.getenv("SIP_PORT", "5060"))

    params = FreeSwitchSIPParams(
        sip_listen_host=sip_host,
        sip_listen_port=sip_port,
    )

    server = FreeSwitchSIPServerTransport(params=params)

    @server.event_handler("on_call_started")
    async def on_call(server, call_transport: FreeSwitchSIPCallTransport):
        logger.info("Call started: %s", call_transport.session.call_id)

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant in a phone call. Your output will be "
                "spoken aloud, so avoid special characters that can't easily be spoken. "
                "Keep responses concise and conversational.",
            },
        ]

        context = LLMContext(messages)
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
        )

        pipeline = Pipeline(
            [
                call_transport.input(),
                stt,
                user_aggregator,
                llm,
                tts,
                call_transport.output(),
                assistant_aggregator,
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        # Kick off the conversation when the call connects
        messages.append({"role": "system", "content": "Please introduce yourself to the caller."})
        await task.queue_frames([LLMRunFrame()])

        runner = PipelineRunner()
        await runner.run(task)

        logger.info("Call ended: %s", call_transport.session.call_id)

    @server.event_handler("on_call_ended")
    async def on_call_ended(server, call_transport: FreeSwitchSIPCallTransport):
        logger.info("Call ended (BYE): %s", call_transport.session.call_id)

    await server.start()
    logger.info("SIP agent listening on %s:%d", sip_host, sip_port)

    # Run until interrupted
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
