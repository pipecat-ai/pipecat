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
from pipecat.observers.startup_timing_observer import StartupTimingObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
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

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
        ]
    )

    latency_observer = UserBotLatencyObserver()
    startup_observer = StartupTimingObserver()

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=[latency_observer, startup_observer],
    )

    @latency_observer.event_handler("on_first_bot_speech_latency")
    async def on_first_bot_speech_latency(observer, latency_seconds):
        logger.info(f"First bot speech: {latency_seconds:.3f}s after client connected")

    @latency_observer.event_handler("on_latency_measured")
    async def on_latency_measured(observer, latency_seconds):
        logger.info(f"⏱️ User-to-bot latency: {latency_seconds:.3f}s")

    @startup_observer.event_handler("on_startup_timing_report")
    async def on_startup_timing_report(observer, report):
        logger.info(f"Total startup: {report.total_duration_secs:.3f}s")
        for timing in report.processor_timings:
            logger.info(f"  {timing.processor_name}: {timing.duration_secs:.3f}s")

    @startup_observer.event_handler("on_transport_timing_report")
    async def on_transport_timing_report(observer, report):
        if report.bot_connected_secs is not None:
            logger.info(f"Bot connected: {report.bot_connected_secs:.3f}s")
        logger.info(f"Client connected: {report.client_connected_secs:.3f}s")

    turn_observer = task.turn_tracking_observer
    if turn_observer:

        @turn_observer.event_handler("on_turn_started")
        async def on_turn_started(observer, turn_number):
            logger.info(f"🔄 Turn {turn_number} started")

        @turn_observer.event_handler("on_turn_ended")
        async def on_turn_ended(observer, turn_number, duration, was_interrupted):
            if was_interrupted:
                logger.info(f"🔄 Turn {turn_number} interrupted after {duration:.2f}s")
            else:
                logger.info(f"🏁 Turn {turn_number} completed in {duration:.2f}s")

    @latency_observer.event_handler("on_latency_breakdown")
    async def on_latency_breakdown(observer, breakdown):
        # Display a sequential waterfall that roughly adds up to the total.
        # User turn is the first stage: user silence → turn release.
        # The STT TTFB is shown as context within the user turn since
        # it's a component of that time (along with VAD silence and any
        # turn analyzer delay).
        stt_ttfb = next((t for t in breakdown.ttfb if "STT" in t.processor), None)
        if breakdown.user_turn_secs is not None:
            stt_note = f" (STT: {stt_ttfb.value:.3f}s)" if stt_ttfb else ""
            logger.info(f"  User turn: {breakdown.user_turn_secs:.3f}s{stt_note}")

        for ttfb in breakdown.ttfb:
            if ttfb is not stt_ttfb:
                logger.info(f"  {ttfb.processor}: TTFB {ttfb.value:.3f}s")

        if breakdown.text_aggregation:
            ta = breakdown.text_aggregation
            logger.info(f"  {ta.processor}: text aggregation {ta.value:.3f}s")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
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
