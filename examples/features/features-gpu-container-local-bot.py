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
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import (
    DailyOutputTransportMessageFrame,
    DailyOutputTransportMessageUrgentFrame,
    DailyParams,
)
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

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = DeepgramTTSService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        settings=DeepgramTTSService.Settings(
            voice="aura-asteria-en",
        ),
        base_url="http://0.0.0.0:8080",
    )

    llm = OpenAILLMService(
        # To use OpenAI
        # api_key=os.environ["OPENAI_API_KEY"],
        # Or, to use a local vLLM (or similar) api server
        settings=OpenAILLMService.Settings(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            system_instruction="You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way.",
        ),
        base_url="http://0.0.0.0:8000/v1",
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            user_aggregator,
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
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

    # When the first participant joins, the bot should introduce itself.
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await task.queue_frames([LLMRunFrame()])

    # Handle "latency-ping" messages. The client will send app messages that look like
    # this:
    #   { "latency-ping": { ts: <client-side timestamp> }}
    #
    # We want to send an immediate pong back to the client from this handler function.
    # Also, we will push a frame into the top of the pipeline and send it after the
    #
    @transport.event_handler("on_app_message")
    async def on_app_message(transport, message, sender):
        try:
            if "latency-ping" in message:
                logger.debug(f"Received latency ping app message: {message}")
                ts = message["latency-ping"]["ts"]
                # Send immediately
                await task.queue_frame(
                    DailyOutputTransportMessageUrgentFrame(
                        message={"latency-pong-msg-handler": {"ts": ts}}, participant_id=sender
                    )
                )
                # And push to the pipeline for the Daily transport.output to send
                await task.queue_frame(
                    DailyOutputTransportMessageFrame(
                        message={"latency-pong-pipeline-delivery": {"ts": ts}},
                        participant_id=sender,
                    )
                )
        except Exception as e:
            logger.debug(f"message handling error: {e} - {message}")

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
