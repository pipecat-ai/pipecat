#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.azure.llm import AzureLLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.speechmatics.stt import DiarizationConfig, SpeechmaticsSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

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


async def run_example(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
    """Run example using Speechmatics STT.

    This example will use diarization within our STT service and output the words spoken by
    each individual speaker and wrap them with XML tags for the LLM to process. Note the
    instructions in the system context for the LLM. This greatly improves the conversation
    experience by allowing the LLM to understand who is speaking in a multi-party call.

    Using the `enable_vad` parameter will use the detection of speakers to emit speaker
    started and stopped frames. This is useful when only wanting to listen to the first
    speaker. Using `focus_speakers` will result in only words from those speakers being
    processed as part of the conversation. Other speakers will be wrapped in PASSIVE tags
    and not trigger any responses from the LLM until words from a focussed speaker have
    been transcribed.

    To use the module's VAD, you must remove `vad_analyzer` from the transport config.

    By default, this example will use our ENHANCED operating point, which is optimized for
    high accuracy. You can change this by setting the `operating_point` parameter to a different
    value.

    For more information on operating points, see the Speechmatics documentation:
    https://docs.speechmatics.com/rt-api-ref"""
    logger.info(f"Starting bot")

    stt = SpeechmaticsSTTService(
        api_key=os.getenv("SPEECHMATICS_API_KEY"),
        language=Language.EN,
        enable_vad=True,
        end_of_utterance_silence_trigger=0.5,
        speaker_active_format="<{speaker_id}>{text}</{speaker_id}>",
        speaker_passive_format="<PASSIVE><{speaker_id}>{text}</{speaker_id}></PASSIVE>",
        diarization_config=DiarizationConfig(
            enable=True,
            max_speakers=10,
            focus_speakers=["S1"],
        ),
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        model="eleven_turbo_v2_5",
    )

    llm = AzureLLMService(
        api_key=os.getenv("AZURE_CHATGPT_API_KEY"),
        endpoint=os.getenv("AZURE_CHATGPT_ENDPOINT"),
        model=os.getenv("AZURE_CHATGPT_MODEL"),
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

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(
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
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Say a short hello to the user."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    main(run_example, transport_params=transport_params)
