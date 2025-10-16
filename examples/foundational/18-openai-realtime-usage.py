#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Example: Print OpenAI Realtime API Token Usage Statistics

This example demonstrates how to access and print token usage statistics
from the OpenAI Realtime API, including detailed breakdowns of input/output
tokens, cached tokens, and audio/text token usage.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

# We store functions so objects don't get instantiated until the desired
# transport gets selected.
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
    """Main function demonstrating usage statistics tracking."""
    logger.info(f"Starting bot")

    # Initialize the OpenAI Realtime service
    llm = OpenAIRealtimeLLMService(
        api_key=os.getenv("OPENAI_API_KEY") or "",
        model="gpt-4o-realtime-preview-2024-12-17",
    )

    # To access usage statistics, we wrap the internal response handler
    # This is the cleanest way to intercept usage data from the realtime API
    original_handler = llm._handle_evt_response_done

    async def custom_response_done_handler(evt):
        """Custom handler that prints usage stats before calling original handler."""
        # Print usage statistics if available
        if evt.response.usage:
            usage = evt.response.usage

            logger.info("\n" + "=" * 50)
            logger.info("📊 TOKEN USAGE STATISTICS")
            logger.info("=" * 50)
            logger.info(f"Total tokens: {usage.total_tokens}")
            logger.info(f"Input tokens: {usage.input_tokens}")
            logger.info(f"Output tokens: {usage.output_tokens}")

            # Input token details
            if usage.input_token_details:
                logger.info(f"\n📥 Input token breakdown:")
                logger.info(f"  • Cached tokens: {usage.input_token_details.cached_tokens}")
                logger.info(f"  • Text tokens: {usage.input_token_details.text_tokens}")
                logger.info(f"  • Audio tokens: {usage.input_token_details.audio_tokens}")

                # Cached token details if available
                if usage.input_token_details.cached_tokens_details:
                    logger.info(
                        f"  • Cached text tokens: {usage.input_token_details.cached_tokens_details.text_tokens}"
                    )
                    logger.info(
                        f"  • Cached audio tokens: {usage.input_token_details.cached_tokens_details.audio_tokens}"
                    )

            # Output token details
            if usage.output_token_details:
                logger.info(f"\n📤 Output token breakdown:")
                logger.info(f"  • Text tokens: {usage.output_token_details.text_tokens}")
                logger.info(f"  • Audio tokens: {usage.output_token_details.audio_tokens}")

            logger.info("=" * 50 + "\n")

        # Call the original handler to maintain normal functionality
        await original_handler(evt)

    # Replace the handler with our custom one
    llm._handle_evt_response_done = custom_response_done_handler

    # Create pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            llm,
            transport.output(),
        ]
    )

    # Create task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        logger.info("🎤 Speak into your microphone to interact with the assistant")
        logger.info("📊 Usage statistics will be printed after each response")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
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
