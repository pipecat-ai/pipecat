#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

import aiohttp
import datetime
import wave
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def save_audio(audiobuffer):
    if audiobuffer.has_audio():
        merged_audio = audiobuffer.merge_audio_buffers()
        filename = f"conversation_recording{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(audiobuffer._sample_rate)
            wf.writeframes(merged_audio)
        print(f"Merged audio saved to {filename}")
    else:
        print("No audio data to save")


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Chatbot",
            DailyParams(
                audio_out_enabled=True,
                audio_in_enabled=True,
                camera_out_enabled=False,
                vad_enabled=True,
                vad_audio_passthrough=True,
                vad_analyzer=SileroVADAnalyzer(),
                transcription_enabled=True,
                #
                # Spanish
                #
                # transcription_settings=DailyTranscriptionSettings(
                #     language="es",
                #     tier="nova",
                #     model="2-general"
                # )
            ),
        )

        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            #
            # English
            #
            voice_id="cgSgspJ2msm6clMCkdW9",
            aiohttp_session=session,
            #
            # Spanish
            #
            # model="eleven_multilingual_v2",
            # voice_id="gD1IexrzCvsXPHUuT0s3",
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        messages = [
            {
                "role": "system",
                #
                # English
                #
                "content": "You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself. Keep all your response to 12 words or fewer.",
                #
                # Spanish
                #
                # "content": "Eres Chatbot, un amigable y útil robot. Tu objetivo es demostrar tus capacidades de una manera breve. Tus respuestas se convertiran a audio así que nunca no debes incluir caracteres especiales. Contesta a lo que el usuario pregunte de una manera creativa, útil y breve. Empieza por presentarte a ti mismo.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        audiobuffer = AudioBufferProcessor()
        pipeline = Pipeline(
            [
                transport.input(),  # microphone
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                audiobuffer,  # used to buffer the audio in the pipeline
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([LLMMessagesFrame(messages)])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            print(f"Participant left: {participant}")
            await task.queue_frame(EndFrame())
            await save_audio(audiobuffer)

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
