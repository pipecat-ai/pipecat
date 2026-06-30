#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.base_llm_adapter import LLMContextMessage
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame, LLMUpdateSettingsFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    UserTurnStoppedMessage,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.xai.realtime import events
from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_stop import BaseUserTurnStopStrategy
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

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
    logger.info(f"Starting bot")

    llm = GrokRealtimeLLMService(api_key=os.environ["XAI_API_KEY"])

    messages: list[LLMContextMessage] = [
        {
            "role": "system",
            "content": "You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way.",
        },
    ]

    context = LLMContext(messages)
    # It appears that Grok Realtime can sometimes be slow to detect the start
    # of a user's turn; uncomment the below imports and user_params to
    # enable "supplemental" interruptions.
    # from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
    # from pipecat.audio.vad.silero import SileroVADAnalyzer
    # from pipecat.turns.user_turn_strategies import UserTurnStrategies
    # from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        # user_params=LLMUserAggregatorParams(
        #     vad_analyzer=SileroVADAnalyzer(),
        #     user_turn_strategies=UserTurnStrategies(start=[VADUserTurnStartStrategy(
        #         enable_interruptions=True,
        #         enable_user_speaking_frames=False,  # Grok already emits turn frames
        #     )], stop=[]) # Grok already emits turn frames
        # ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            llm,
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

    # Grok emits user-turn frames from server VAD, so
    # on_user_turn_stopped fires at the turn boundary. In realtime mode
    # UserTurnStoppedMessage.content is None (the user transcript isn't
    # finalized at turn-stop time); subscribe to on_user_turn_message_added
    # if you need the finalized user text.
    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(
        aggregator,
        strategy: BaseUserTurnStopStrategy,
        message: UserTurnStoppedMessage,
    ):
        logger.info(f"User turn stopped at {message.timestamp}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}assistant: {message.content}"
        logger.info(f"Transcript: {line}")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        await worker.queue_frames([LLMRunFrame()])

        await asyncio.sleep(10)
        logger.info("Updating Grok Realtime LLM settings: voice='Rex'")
        await worker.queue_frame(
            LLMUpdateSettingsFrame(
                delta=GrokRealtimeLLMService.Settings(
                    session_properties=events.SessionProperties(voice="Rex")
                )
            )
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
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
