#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Audio Recording Example with Pipecat.

This example demonstrates how to record audio from a conversation between a user and an AI assistant,
saving both merged and individual audio tracks. It showcases the AudioBufferProcessor's capabilities
to handle both combined and separate audio streams.

The example:
    1. Sets up a basic conversation with an AI assistant
    2. Records the entire conversation
    3. Saves three separate WAV files:
        - A merged recording of both participants
        - Individual recording of user audio
        - Individual recording of assistant audio

Example usage (run from pipecat root directory):
    $ pip install "pipecat-ai[daily,openai,cartesia,silero]"
    $ pip install -r dev-requirements.txt
    $ python examples/foundational/34-audio-recording.py

Requirements:
    - OpenAI API key (for GPT-4)
    - Cartesia API key (for text-to-speech)
    - Daily API key (for video/audio transport)

    Environment variables (.env file):
        OPENAI_API_KEY=your_openai_key
        CARTESIA_API_KEY=your_cartesia_key
        DAILY_API_KEY=your_daily_key
        DEEPGRAM_API_KEY=your_deepgram_key

The recordings will be saved in a 'recordings' directory with timestamps:
    recordings/
        merged_20240315_123456.wav  (Combined audio)
        user_20240315_123456.wav    (User audio only)
        bot_20240315_123456.wav     (Bot audio only)

Note:
    This example requires the AudioBufferProcessor with track-specific audio support,
    which provides both 'on_audio_data' and 'on_track_audio_data' events for
    handling merged and separate audio tracks respectively.
"""

import asyncio
import datetime
import io
import os
import sys
import wave

import aiofiles
import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def save_audio_file(audio: bytes, filename: str, sample_rate: int, num_channels: int):
    """Save audio data to a WAV file."""
    if len(audio) > 0:
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())
        logger.info(f"Audio saved to {filename}")


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Recording bot",
            DailyParams(
                # audio_in_enabled=True,
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"), audio_passthrough=True)

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")

        # Create audio buffer processor
        audiobuffer = AudioBufferProcessor()

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant demonstrating audio recording capabilities. Keep your responses brief and clear.",
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
                audiobuffer,  # Add audio buffer to pipeline
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            await audiobuffer.start_recording()
            messages.append(
                {
                    "role": "system",
                    "content": "Greet the user and explain that this conversation will be recorded.",
                }
            )
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await audiobuffer.stop_recording()
            await task.cancel()

        # Handler for merged audio
        @audiobuffer.event_handler("on_audio_data")
        async def on_audio_data(buffer, audio, sample_rate, num_channels):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recordings/merged_{timestamp}.wav"
            os.makedirs("recordings", exist_ok=True)
            await save_audio_file(audio, filename, sample_rate, num_channels)

        # Handler for separate tracks
        @audiobuffer.event_handler("on_track_audio_data")
        async def on_track_audio_data(buffer, user_audio, bot_audio, sample_rate, num_channels):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("recordings", exist_ok=True)

            # Save user audio
            user_filename = f"recordings/user_{timestamp}.wav"
            await save_audio_file(user_audio, user_filename, sample_rate, 1)

            # Save bot audio
            bot_filename = f"recordings/bot_{timestamp}.wav"
            await save_audio_file(bot_audio, bot_filename, sample_rate, 1)

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
