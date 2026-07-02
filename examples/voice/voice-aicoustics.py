#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import datetime
import os
import wave

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.filters.aic_filter import AICFilter
from pipecat.audio.vad.aic_quail_vad import AICQuailVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


def _create_aic_filter() -> AICFilter:
    license_key = os.environ["AIC_SDK_LICENSE"]

    return AICFilter(
        license_key=license_key,
        model_id="quail-vf-2.2-l-16khz",
        enhancement_level=0.8,
    )


aic_filter = _create_aic_filter()
aic_vad_analyzer = AICQuailVADAnalyzer(
    license_key=os.environ["AIC_SDK_LICENSE"],
)

# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=aic_filter,
    ),
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=aic_filter,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=aic_filter,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=aic_filter,
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
        user_params=LLMUserAggregatorParams(vad_analyzer=aic_vad_analyzer),
    )

    # Create audio buffer processor so we can hear the audio fitler results.
    audiobuffer = AudioBufferProcessor(
        num_channels=2,  # 1 for mono, 2 for stereo (user left, bot right)
        enable_turn_audio=False,  # Enable per-turn audio recording
        auto_start_recording=True,  # Start recording automatically when the pipeline starts
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            audiobuffer,  # write audio data to a file
            assistant_aggregator,  # Assistant spoken responses
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

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @audiobuffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels):
        # Save or process the composite audio
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./conversation_{timestamp}.wav"

        # Create the WAV file
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(audio)

        logger.info(f"Saved recording to {filename}")

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
