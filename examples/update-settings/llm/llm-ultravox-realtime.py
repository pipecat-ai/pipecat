#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import datetime
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.base_llm_adapter import LLMContextMessage
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame, LLMUpdateSettingsFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.ultravox.llm import OneShotInputParams, UltravoxRealtimeLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
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

    system_prompt = "You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way."

    llm = UltravoxRealtimeLLMService(
        params=OneShotInputParams(
            api_key=os.environ["ULTRAVOX_API_KEY"],
            system_prompt=system_prompt,
            temperature=0.3,
            max_duration=datetime.timedelta(minutes=3),
        ),
        one_shot_selected_tools=ToolsSchema(standard_tools=[]),
    )

    messages: list[LLMContextMessage] = [
        {
            "role": "system",
            "content": system_prompt,
        },
    ]

    context = LLMContext(messages)
    # Ultravox doesn't emit user-turn frames. To get them (for RTVI
    # speech events, turn observers, etc.) uncomment the local-VAD
    # imports + `user_params=` below. See realtime-ultravox.py for the
    # full discussion.
    #
    # from pipecat.audio.vad.silero import SileroVADAnalyzer
    # from pipecat.processors.aggregators.llm_response_universal import (
    #     LLMUserAggregatorParams,
    # )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        realtime_service_mode=True,
        # user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
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

    # Ultravox doesn't emit user-turn frames, so on_user_turn_stopped
    # won't fire. If you uncomment the local-VAD opt-in above, also
    # uncomment the imports and handler below.
    #
    # from pipecat.processors.aggregators.llm_response_universal import UserTurnStoppedMessage
    # from pipecat.turns.user_stop import BaseUserTurnStopStrategy
    #
    # @user_aggregator.event_handler("on_user_turn_stopped")
    # async def on_user_turn_stopped(
    #     aggregator,
    #     strategy: BaseUserTurnStopStrategy,
    #     message: UserTurnStoppedMessage,
    # ):
    #     logger.info(f"User turn stopped at {message.timestamp}")

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
        logger.info("Updating Ultravox Realtime LLM settings: output_medium=text")
        await worker.queue_frame(
            LLMUpdateSettingsFrame(delta=UltravoxRealtimeLLMService.Settings(output_medium="text"))
        )

        await asyncio.sleep(10)
        logger.info("Updating Ultravox Realtime LLM settings: output_medium=voice")
        await worker.queue_frame(
            LLMUpdateSettingsFrame(delta=UltravoxRealtimeLLMService.Settings(output_medium="voice"))
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
