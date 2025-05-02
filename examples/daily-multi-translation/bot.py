#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.audio.mixers.soundfile_mixer import SoundfileMixer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

BACKGROUND_SOUND_FILE = "office-ambience-mono-16000.mp3"


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Multi translation bot",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_out_mixer={
                    "spanish": SoundfileMixer(
                        sound_files={"office": BACKGROUND_SOUND_FILE}, default_sound="office"
                    ),
                    "french": SoundfileMixer(
                        sound_files={"office": BACKGROUND_SOUND_FILE}, default_sound="office"
                    ),
                    "german": SoundfileMixer(
                        sound_files={"office": BACKGROUND_SOUND_FILE}, default_sound="office"
                    ),
                },
                audio_out_destinations=["spanish", "french", "german"],
                microphone_out_enabled=False,  # Disable since we just use custom tracks
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        tts_spanish = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="cefcb124-080b-4655-b31f-932f3ee743de",
            transport_destination="spanish",
        )
        tts_french = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="8832a0b5-47b2-4751-bb22-6a8e2149303d",
            transport_destination="french",
        )
        tts_german = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="38aabb6a-f52b-4fb0-a3d1-988518f4dc06",
            transport_destination="german",
        )

        messages_spanish = [
            {
                "role": "system",
                "content": "You will be provided with a sentence in English, and your task is to only translate it into Spanish.",
            },
        ]
        messages_french = [
            {
                "role": "system",
                "content": "You will be provided with a sentence in English, and your task is to only translate it into French.",
            },
        ]
        messages_german = [
            {
                "role": "system",
                "content": "You will be provided with a sentence in English, and your task is to only translate it into German.",
            },
        ]

        llm_spanish = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
        llm_french = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
        llm_german = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

        context_spanish = OpenAILLMContext(messages_spanish)
        context_aggregator_spanish = llm_spanish.create_context_aggregator(context_spanish)

        context_french = OpenAILLMContext(messages_french)
        context_aggregator_french = llm_french.create_context_aggregator(context_french)

        context_german = OpenAILLMContext(messages_german)
        context_aggregator_german = llm_german.create_context_aggregator(context_german)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt,
                ParallelPipeline(
                    # Spanish pipeline.
                    [
                        context_aggregator_spanish.user(),
                        llm_spanish,
                        tts_spanish,
                        context_aggregator_spanish.assistant(),
                    ],
                    # French pipeline.
                    [
                        context_aggregator_french.user(),
                        llm_french,
                        tts_french,
                        context_aggregator_french.assistant(),
                    ],
                    # German pipeline.
                    [
                        context_aggregator_german.user(),
                        llm_german,
                        tts_german,
                        context_aggregator_german.assistant(),
                    ],
                ),
                transport.output(),  # Transport bot output
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=16000,
                audio_out_sample_rate=16000,
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                report_only_initial_ttfb=True,
            ),
            observers=[TranscriptionLogObserver()],
        )

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
