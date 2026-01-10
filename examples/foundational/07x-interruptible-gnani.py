#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Example demonstrating Gnani STT service integration with Pipecat.

This example shows how to use Gnani's multilingual speech-to-text service
in a conversational AI pipeline with interruption support.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.openai_llm_context import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.gnani.stt import GnaniSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.turn_strategies import (
    TurnAnalyzerBotTurnStartStrategy,
    TurnStartStrategies,
)
from pipecat.turns.turn_utils import LocalSmartTurnAnalyzerV3
from pipecat.universal_context.llm_context import LLMContext

load_dotenv(override=True)


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot with Gnani STT")

    # Initialize Gnani STT service with API credentials
    # You can set the language to any supported Indian language
    from pipecat.transcriptions.language import Language

    stt = GnaniSTTService(
        api_key=os.getenv("GNANI_API_KEY"),
        organization_id=os.getenv("GNANI_ORGANIZATION_ID"),
        params=GnaniSTTService.InputParams(
            language=Language.HI_IN,  # Hindi by default, change as needed
            api_user_id=os.getenv("GNANI_USER_ID", "pipecat-user"),
        ),
    )

    # TTS service
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    # LLM service
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    # System message for the LLM
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant that can understand and respond in multiple Indian languages. "
            "Your goal is to demonstrate your capabilities in a succinct way. "
            "Your output will be spoken aloud, so avoid special characters that can't easily be spoken. "
            "Respond to what the user says in a creative and helpful way.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            turn_start_strategies=TurnStartStrategies(
                bot=[TurnAnalyzerBotTurnStartStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # Gnani STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # Register an event handler for when a client connects
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # You can greet the user when they connect
        # await task.queue_frames([TTSSpeakFrame("Hello! I'm using Gnani STT service. How can I help you?")])

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    import asyncio
    from pipecat.runner import main as run

    asyncio.run(run(bot))

