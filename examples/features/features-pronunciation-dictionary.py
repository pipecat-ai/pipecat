#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Say medication names right with a pronunciation dictionary.

Loads a pronunciation dictionary exported by pipecat-tts-pronunciation-evals
(the dashboard's "Export pronunciations") and gives it to the TTS service through
its pronunciation text transforms. The file holds, for each evaluated service,
the model the pronunciations were measured on and the words to hint:

    {
      "services": {
        "cartesia": {
          "model": "sonic-3.6",
          "ipa": {"Adalimumab": "ˌædəˈlɪmumæb"},
          "arpabet": {},
          "plain": ["Abilify"],
          "unresolved": ["Carisoprodol"]
        }
      }
    }

``pronounce_ipa`` and ``pronounce_arpabet`` are classmethods of every TTS
service, and each service writes the pronunciation in its own markup: Cartesia
turns "ˌædəˈlɪmumæb" into ``<<ˌ|æ|d|ə|l|ˈ|ɪ|m|u|m|æ|b>>``. ``plain`` words are
already said right without a hint, and ``unresolved`` ones were never said
right, so neither is passed on.

The bot is a pharmacy assistant that talks about the medications in the file.

Run locally:
    python features-pronunciation-dictionary.py

Run against a Daily room:
    python features-pronunciation-dictionary.py -t daily

Requires:
    pip install pipecat-ai[cartesia,openai,silero,daily]
"""

import json
import os
from pathlib import Path

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
from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

PRONUNCIATIONS = Path(__file__).parent.parent / "assets" / "pronunciations.json"

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

    dictionary = json.loads(PRONUNCIATIONS.read_text())["services"]["cartesia"]
    hinted = [*dictionary["ipa"], *dictionary["arpabet"]]

    stt = CartesiaSTTService(api_key=os.environ["CARTESIA_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            # The pronunciations were measured on this model and voice.
            model=dictionary["model"],
            voice="86e30c1d-714b-4074-a1f2-1cb6b552fb49",
        ),
        # Each word is matched whole and case-insensitively, wherever the LLM says it.
        text_transforms=[
            ("*", CartesiaTTSService.pronounce_ipa(dictionary["ipa"])),
            ("*", CartesiaTTSService.pronounce_arpabet(dictionary["arpabet"])),
        ],
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=(
                "You are a pharmacy assistant in a voice conversation. Your responses are "
                "spoken aloud, so keep them short and avoid formatting that can't be "
                "spoken. You can talk about these medications, spelled exactly as written "
                f"here: {', '.join(hinted)}."
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
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {
                "role": "developer",
                "content": (
                    "Greet the caller and offer to help with their prescriptions, naming a "
                    f"few of the medications you know, such as {', '.join(hinted[:3])}."
                ),
            }
        )
        await worker.queue_frames([LLMRunFrame()])

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
