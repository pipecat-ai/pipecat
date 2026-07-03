#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice formatting with VoiceFormatter.

Demonstrates the VoiceFormatter bundle, which applies a pipeline of built-in
text transforms before TTS synthesis so that currency amounts, phone numbers,
dates, acronyms, and other special text are spoken naturally.

Without voice formatting a TTS service might read:
  "$42.50" as "dollar sign four two point five zero"
  "API"    as a single word rather than "A P I"
  "3/15"   as "three slash fifteen"

With VoiceFormatter these are pre-processed before the audio is synthesised:
  "$42.50"  →  "forty-two dollars and fifty cents"
  "API"     →  "A P I"
  "3/15/25" →  "March 15th, two thousand and twenty-five"

Run locally:
    python features-voice-formatter.py

Run against a Daily room:
    python features-voice-formatter.py -t daily

Requires:
    pip install pipecat-ai[cartesia,deepgram,openai,silero,daily]
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.utils.text.transforms import VoiceFormatter
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
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

    stt = CartesiaSTTService(api_key=os.environ["CARTESIA_API_KEY"])

    # VoiceFormatter with all defaults enabled.
    # Pass explicit flags to turn individual transforms on or off, e.g.:
    #   VoiceFormatter(expand_numbers=True, normalize_acronyms=False)
    voice_formatter = VoiceFormatter()

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        settings=ElevenLabsTTSService.Settings(
            voice=os.getenv("ELEVENLABS_VOICE_ID", ""),
        ),
        # Attach VoiceFormatter as a text transform. The "*" aggregation type
        # means it runs on every text frame regardless of how it was aggregated.
        text_transforms=[("*", voice_formatter)],
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=(
                "You are a billing support assistant for a telecom company. "
                "Your responses are spoken aloud. Use natural formatting in your "
                "answers: currency amounts like $42.50, percentages like 3.5%, "
                "email addresses like support@example.com, and abbreviations like "
                "Dr. or St. as you normally would in writing — the voice formatter "
                "will convert them to natural speech before synthesis."
            ),
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {
                "role": "developer",
                "content": (
                    "Greet the user and let them know they've reached billing support. "
                    "Offer to help with their account balance, recent charges, or "
                    "payment options. Give a sample balance such as $127.50 due on "
                    "3/15/2025 and a support email like billing@telecom.example.com."
                ),
            }
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
