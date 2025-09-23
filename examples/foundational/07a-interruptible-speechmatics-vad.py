#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response import (
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.speechmatics.stt import SpeechmaticsSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
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
    """Speechmatics STT Service Example

    This example demonstrates using Speechmatics Speech-to-Text service with speaker diarization and intelligent speaker management. Key features:

    1. Speaker Diarization
       - Automatically identifies and distinguishes between different speakers
       - First speaker is identified as 'S1', others get subsequent IDs
       - Uses `enable_diarization` parameter to manage speaker detection

    2. Smart Speaker Control
       - `focus_speakers` parameter lets you target specific speakers (e.g. ["S1"])
       - Other speakers will be wrapped in PASSIVE tags
       - Only processes speech from focused speakers
       - Words from all speakers are wrapped with XML tags for clear speaker identification
       - Other speakers' speech only sent when focused speaker is active

    3. Voice Activity Detection
       - Built-in VAD using `enable_vad` parameter
       - Remove `vad_analyzer` from `transport` config to use module's VAD
       - Emits speaker started/stopped events

    4. Configuration Options
       - `operating_point` parameter defaults to `ENHANCED` for optimal accuracy
       - Configurable `end_of_utterance_silence_trigger` (default 0.5s)
       - Customizable speaker formatting
       - Additional diarization settings available

    For detailed information about operating points and configuration:
    https://docs.speechmatics.com/rt-api-ref
    """

    logger.info(f"Starting bot")

    stt = SpeechmaticsSTTService(
        api_key=os.getenv("SPEECHMATICS_API_KEY"),
        params=SpeechmaticsSTTService.InputParams(
            language=Language.EN,
            enable_vad=True,
            enable_diarization=True,
            focus_speakers=["S1"],
            end_of_utterance_silence_trigger=0.5,
            speaker_active_format="<{speaker_id}>{text}</{speaker_id}>",
            speaker_passive_format="<PASSIVE><{speaker_id}>{text}</{speaker_id}></PASSIVE>",
        ),
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        model="eleven_turbo_v2_5",
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        params=BaseOpenAILLMService.InputParams(temperature=0.75),
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful British assistant called Alfred. "
                "Your goal is to demonstrate your capabilities in a succinct way. "
                "Your output will be converted to audio so don't include special characters in your answers. "
                "Always include punctuation in your responses. "
                "Give very short replies - do not give longer replies unless strictly necessary. "
                "Respond to what the user said in a concise, funny, creative and helpful way. "
                "Use `<Sn/>` tags to identify different speakers - do not use tags in your replies. "
                "Do not respond to speakers within `<PASSIVE/>` tags unless explicitly asked to. "
            ),
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(aggregation_timeout=0.005),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
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
        messages.append({"role": "system", "content": "Say a short hello to the user."})
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
