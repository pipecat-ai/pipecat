#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import os
import random
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    UserTurnMessageAddedMessage,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService
from pipecat.services.aws.nova_sonic.session_continuation import SessionContinuationParams
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

# Load environment variables
load_dotenv(override=True)


async def get_current_weather(params: FunctionCallParams, location: str, format: str):
    """Get the current weather.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
        format: The temperature unit to use. Must be either "celsius" or "fahrenheit". Infer this from the user's location.
    """
    temperature = random.randint(60, 85) if format == "fahrenheit" else random.randint(15, 30)
    await params.result_callback(
        {
            "conditions": "nice",
            "temperature": temperature,
            "location": location,
            "format": format,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
    )


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
    logger.info(f"Starting bot")

    # Specify initial system instruction.
    system_instruction = (
        "You are a friendly assistant. The user and you will engage in a spoken dialog exchanging "
        "the transcripts of a natural real-time conversation. Keep your responses short, generally "
        "two or three sentences for chatty scenarios."
        # HACK: if using the older Nova Sonic (pre-2) model, note that you need to inject a special
        # bit of text into this instruction to allow the first assistant response to be
        # programmatically triggered (which happens in the on_client_connected handler)
        # f"{AWSNovaSonicLLMService.AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION}"
    )

    # Create the AWS Nova Sonic LLM service
    llm = AWSNovaSonicLLMService(
        secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        # as of 2025-12-09, these are the supported regions:
        # - Nova 2 Sonic (the default model):
        #   - us-east-1
        #   - us-west-2
        #   - ap-northeast-1
        # - Nova Sonic (the older model):
        #   - us-east-1
        #   - ap-northeast-1
        region=os.environ["AWS_REGION"],
        session_token=os.getenv("AWS_SESSION_TOKEN"),
        settings=AWSNovaSonicLLMService.Settings(
            voice="tiffany",
            system_instruction=system_instruction,
        ),
        # Session continuation is enabled by default, allowing seamless
        # conversations longer than the AWS ~8-minute session limit.
        # The service rotates sessions in the background with no
        # user-perceptible interruption. You can tune the threshold or
        # disable it with: session_continuation=SessionContinuationParams(enabled=False)
        session_continuation=SessionContinuationParams(
            # When to start preparing the next session (default: 360 = 6 min).
            # Lower this (e.g. 20) to see a handoff happen quickly during testing.
            transition_threshold_seconds=360,
        ),
        # you could choose to pass tools here rather than via context
        # tools=[get_current_weather]
    )

    # AWS Nova Sonic drives the conversation server-side.
    #
    # It does not, however, emit turn frames (UserStartedSpeakingFrame,
    # UserStoppedSpeakingFrame). realtime_service_mode ensures that context
    # aggregation will work without those frames, but you can add supplemental
    # local turn frames for consumption by other pipeline processors that
    # expect them (like RTVI), or to trigger on_user_turn_* events. WARNING:
    # you should consider supplemental local turn frames approximate, as they
    # may not always align with server turns.
    #
    # To enable supplemental local turn frames, uncomment the SileroVADAnalyzer
    # and related imports below and the `user_params=` argument further down.
    # Doing so enables the on_user_turn_stopped event, which you could then
    # also uncomment.
    #
    # from pipecat.audio.vad.silero import SileroVADAnalyzer
    # from pipecat.processors.aggregators.llm_response_universal import (
    #     LLMUserAggregatorParams,
    #     UserTurnStoppedMessage,
    # )
    # from pipecat.turns.user_stop import BaseUserTurnStopStrategy

    context = LLMContext(tools=[get_current_weather])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        realtime_service_mode=True,
        # user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    # Build the pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            llm,
            transport.output(),
            assistant_aggregator,
        ]
    )

    # Configure the pipeline worker
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # Handle client connection event
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])
        # HACK: if using the older Nova Sonic (pre-2) model, you need this special way of
        # triggering the first assistant response. Note that this trigger requires a special
        # corresponding bit of text in the system instruction.
        # await llm.trigger_assistant_response()

    # Handle client disconnection events
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await worker.cancel()

    # See comment above the user_aggregator for details on why this is
    # commented out and instructions for enabling it.
    # @user_aggregator.event_handler("on_user_turn_stopped")
    # async def on_user_turn_stopped(
    #     aggregator,
    #     strategy: BaseUserTurnStopStrategy,
    #     message: UserTurnStoppedMessage,
    # ):
    #     logger.info(f"User turn stopped at {message.timestamp}")

    @user_aggregator.event_handler("on_user_turn_message_added")
    async def on_user_turn_message_added(aggregator, message: UserTurnMessageAddedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}user: {message.content}"
        logger.info(f"Transcript: {line}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}assistant: {message.content}"
        logger.info(f"Transcript: {line}")

    # Run the pipeline
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
