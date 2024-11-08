#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
import uuid

import aiohttp
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
from pipecat.services.canonical import CanonicalMetricsService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


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
                "content": "You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself. Keep all your responses to 12 words or fewer.",
                #
                # Spanish
                #
                # "content": "Eres Chatbot, un amigable y útil robot. Tu objetivo es demostrar tus capacidades de una manera breve. Tus respuestas se convertiran a audio así que nunca no debes incluir caracteres especiales. Contesta a lo que el usuario pregunte de una manera creativa, útil y breve. Empieza por presentarte a ti mismo.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        """
        CanonicalMetrics uses AudioBufferProcessor under the hood to buffer the audio. On
        call completion, CanonicalMetrics will send the audio buffer to Canonical for
        analysis. Visit https://voice.canonical.chat to learn more.
        """
        audio_buffer_processor = AudioBufferProcessor()
        canonical = CanonicalMetricsService(
            audio_buffer_processor=audio_buffer_processor,
            aiohttp_session=session,
            api_key=os.getenv("CANONICAL_API_KEY"),
            api_url=os.getenv("CANONICAL_API_URL"),
            call_id=str(uuid.uuid4()),
            assistant="pipecat-chatbot",
            assistant_speaks_first=True,
        )
        pipeline = Pipeline(
            [
                transport.input(),  # microphone
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                audio_buffer_processor,  # captures audio into a buffer
                canonical,  # uploads audio buffer to Canonical AI for metrics
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

        @transport.event_handler("on_call_state_updated")
        async def on_call_state_updated(transport, state):
            if state == "left":
                await task.queue_frame(EndFrame())

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
