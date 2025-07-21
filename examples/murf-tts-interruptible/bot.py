# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.murf.tts import MurfTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    settings = {
        "murf_api_key": os.getenv("MURF_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "deepgram_api_key": os.getenv("DEEPGRAM_API_KEY"),
    }

    logger.info(f"Settings: {settings}")

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        )
    )

    stt = DeepgramSTTService(api_key=settings["deepgram_api_key"])

    tts = MurfTTSService(
        api_key=settings["murf_api_key"],
        params=MurfTTSService.InputParams(
            voice_id="en-US-daniel",
            style="Conversational",
            rate=0,
            pitch=0,
            variation=1,
            sample_rate=44100,
            channel_type="MONO",
            format="PCM",
        ),
    )

    llm = OpenAILLMService(api_key=settings["openai_api_key"])

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(), 
            stt,
            context_aggregator.user(),  
            llm,  
            tts,  
            transport.output(),  
            context_aggregator.assistant(),  
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    messages.append({"role": "system", "content": "Please introduce yourself to the user."})
    await task.queue_frames([context_aggregator.user().get_context_frame()])

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    logger.info("Starting bot")
    asyncio.run(main())
