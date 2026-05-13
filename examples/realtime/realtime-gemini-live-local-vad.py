#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, GeminiVADParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
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

    llm = GeminiLiveLLMService(
        api_key=os.environ["GOOGLE_API_KEY"],
        settings=GeminiLiveLLMService.Settings(
            voice="Aoede",  # Puck, Charon, Kore, Fenrir, Aoede
            vad=GeminiVADParams(disabled=True),
        ),
        # inference_on_context_initialization=False,
    )

    context = LLMContext(
        [
            {
                "role": "user",
                "content": "Say hello. Then ask if I want to hear a joke.",
            },
        ],
    )
    # `wait_for_transcript_to_end_user_turn=False` configures the user
    # aggregator for realtime services like Gemini Live that emit user
    # transcripts after the audible end of the turn. With this flag the
    # aggregator:
    #
    # - drops `TranscriptionUserTurnStartStrategy` from the default start
    #   strategies (so late-arriving realtime transcripts don't trigger
    #   new turns),
    # - sets `wait_for_transcript=False` on the default stop strategy
    #   (so the turn ends without waiting for a transcript),
    # - fires `on_user_turn_stopped` on the audible end of the turn with
    #   empty `message.content` (the transcript hasn't arrived yet), and
    # - defers the context flush until the (late) transcript arrives, then
    #   emits `on_user_turn_message_finalized` with the populated message
    #   so the user's words land in the LLM context for audit/history.
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            wait_for_transcript_to_end_user_turn=False,
        ),
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

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    # With `wait_for_transcript_to_end_user_turn=False`, `on_user_turn_stopped`
    # fires on the audible end of the turn (before the transcript arrives), so
    # its `message.content` is empty. Logged here to make the timing of the
    # audible-stop signal visible alongside the later transcript event.
    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        logger.info(f"User turn ended (audible, strategy: {type(strategy).__name__})")

    # `on_user_turn_message_finalized` fires when the user transcript has
    # been written to context — later than `on_user_turn_stopped` in this
    # mode, since transcripts arrive after the audible turn end.
    @user_aggregator.event_handler("on_user_turn_message_finalized")
    async def on_user_turn_message_finalized(aggregator, strategy, message: UserTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}user: {message.content}"
        logger.info(f"Transcript: {line}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}assistant: {message.content}"
        logger.info(f"Transcript: {line}")

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
