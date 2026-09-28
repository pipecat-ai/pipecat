#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.speechmatics.stt import SpeechmaticsSTTService
from pipecat.services.speechmatics.tts import SpeechmaticsTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

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
    """Speechmatics STT and TTS Service Example (server-side VAD).

    This example demonstrates Speechmatics Speech-to-Text and Text-to-Speech with speaker
    diarization, using the service's own VAD for turn detection. Key features:

    1. Speaker Diarization (STT)
       - Automatically identifies and distinguishes between different speakers
       - First speaker is identified as 'S1', others get subsequent IDs
       - Enabled with the `enable_diarization` parameter
       - `speaker_active_format` wraps each result with the speaker label for the LLM

    2. Turn detection (STT)
       - `turn_detection_mode=VAD`: the Speechmatics service runs its own VAD and closes
         turns itself, so no external VAD is needed in the pipeline
       - Use `EXTERNAL` instead to drive turns from Pipecat's own VAD via `finalize()`

    3. Text-to-Speech (TTS)
       - Low latency streaming audio synthesis
       - Multiple voice options available including `sarah`, `theo`, `megan` and `jack`

    For detailed information:
    - STT: https://docs.speechmatics.com/rt-api-ref
    - TTS: https://docs.speechmatics.com/text-to-speech/quickstart
    """

    logger.info("Starting bot")
    async with aiohttp.ClientSession() as session:
        stt = SpeechmaticsSTTService(
            api_key=os.environ["SPEECHMATICS_API_KEY"],
            settings=SpeechmaticsSTTService.Settings(
                language=Language.EN,
                # VAD mode: the Speechmatics service runs its own VAD and closes turns itself.
                turn_detection_mode=SpeechmaticsSTTService.TurnDetectionMode.VAD,
                enable_diarization=True,
                speaker_active_format="<{speaker_id}>{text}</{speaker_id}>",
            ),
        )

        tts = SpeechmaticsTTSService(
            api_key=os.environ["SPEECHMATICS_API_KEY"],
            settings=SpeechmaticsTTSService.Settings(
                voice="sarah",
            ),
            aiohttp_session=session,
        )

        llm = OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMService.Settings(
                temperature=0.75,
                system_instruction="You are a helpful British assistant called Sarah in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Always include punctuation in your responses. Give very short replies - do not give longer replies unless strictly necessary. Respond to what the user said in a concise, funny, creative and helpful way. Use `<Sn/>` tags to identify different speakers - do not use tags in your replies.",
            ),
        )

        context = LLMContext()
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

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

        worker = PipelineWorker(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
            processor_unusable_policy=ProcessorUnusablePolicy.END,
        )

        runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

        await runner.add_workers(worker)

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            # Kick off the conversation.
            context.add_message({"role": "developer", "content": "Say a short hello to the user."})
            await worker.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await runner.cancel()

        await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
