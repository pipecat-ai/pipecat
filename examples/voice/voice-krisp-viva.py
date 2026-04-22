#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Interruptible bot with Krisp VIVA noise filtering, turn detection, and IP.

This example demonstrates a conversational bot with:
- Krisp VIVA noise reduction on incoming audio
- Krisp VIVA Turn detection for end-of-turn
- Krisp Interruption Prediction (IP) to filter backchannels from real interruptions
- Voice activity detection (VAD)

Required environment variables:
- KRISP_VIVA_FILTER_MODEL_PATH: Path to the Krisp noise filter model file (.kef)
- KRISP_VIVA_TURN_MODEL_PATH: Path to the Krisp turn detection model file (.kef)
- KRISP_VIVA_IP_MODEL_PATH: Path to the Krisp IP model file (.kef)
- DEEPGRAM_API_KEY: Deepgram API key for STT
- CARTESIA_API_KEY: Cartesia API key for TTS
- OPENAI_API_KEY: OpenAI API key for LLM

Optional environment variables:
- KRISP_VIVA_API_KEY: Krisp SDK API key (or set in code)
- KRISP_NOISE_SUPPRESSION_LEVEL: Noise suppression level 0-100 (default: 100)
  Higher values = more aggressive noise reduction
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.filters.krisp_viva_filter import KrispVivaFilter
from pipecat.audio.turn.krisp_viva_turn import KrispVivaTurn
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.metrics.metrics import TurnMetricsData
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
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
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_start import (
    KrispVivaIPUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
)
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

load_dotenv(override=True)

transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=KrispVivaFilter(),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=KrispVivaFilter(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=KrispVivaFilter(),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction="You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way.",
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[
                    KrispVivaIPUserTurnStartStrategy(threshold=0.5),
                    TranscriptionUserTurnStartStrategy(),
                ],
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=KrispVivaTurn())],
            ),
            vad_analyzer=SileroVADAnalyzer(),  # or KrispVivaVadAnalyzer
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=[MetricsLogObserver(include_metrics={TurnMetricsData})],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
