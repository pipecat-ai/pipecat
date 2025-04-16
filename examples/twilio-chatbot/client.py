#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import datetime
import io
import os
import sys
import wave
import xml.etree.ElementTree as ET
from uuid import uuid4

import aiofiles
import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import EndFrame, TransportMessageUrgentFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.network.websocket_client import (
    WebsocketClientParams,
    WebsocketClientTransport,
)

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


DEFAULT_CLIENT_DURATION = 30


async def download_twiml(server_url: str) -> str:
    # TODO(aleix): add error checking.
    async with aiohttp.ClientSession() as session:
        async with session.post(server_url) as response:
            return await response.text()


def get_stream_url_from_twiml(twiml: str) -> str:
    root = ET.fromstring(twiml)
    # TODO(aleix): add error checking.
    stream_element = root.find(".//Stream")  # Finds the first <Stream> element
    url = stream_element.get("url")
    return url


async def save_audio(client_name: str, audio: bytes, sample_rate: int, num_channels: int):
    if len(audio) > 0:
        filename = (
            f"{client_name}_recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        )
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())
        logger.info(f"Merged audio saved to {filename}")
    else:
        logger.info("No audio data to save")


async def run_client(client_name: str, server_url: str, duration_secs: int):
    twiml = await download_twiml(server_url)

    stream_url = get_stream_url_from_twiml(twiml)

    stream_sid = str(uuid4())

    transport = WebsocketClientTransport(
        uri=stream_url,
        params=WebsocketClientParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=TwilioFrameSerializer(stream_sid),
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=1.5)),
            vad_audio_passthrough=True,
        ),
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    # We let the audio passthrough so we can record the conversation.
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        audio_passthrough=True,
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="e13cae5c-ec59-4f71-b0a6-266df3c9bb8e",  # Madame Mischief
        push_silence_after_stop=True,
    )

    messages = [
        {
            "role": "system",
            "content": "You are an 8 year old child. A teacher will explain you new concepts you want to know about. Feel free to change topics whnever you want. Once you are taught something you need to keep asking for clarifications if you think someone your age would not understand what you are being taught.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # NOTE: Watch out! This will save all the conversation in memory. You can
    # pass `buffer_size` to get periodic callbacks.
    audiobuffer = AudioBufferProcessor(user_continuous_stream=False)

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from server
            stt,  # Speech-To-Text
            context_aggregator.user(),
            llm,  # LLM
            tts,  # Text-To-Speech
            transport.output(),  # Websocket output to server
            audiobuffer,  # Used to buffer the audio in the pipeline
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            allow_interruptions=True,
        ),
    )

    @transport.event_handler("on_connected")
    async def on_connected(transport: WebsocketClientTransport, client):
        # Start recording.
        await audiobuffer.start_recording()

        message = TransportMessageUrgentFrame(
            message={"event": "connected", "protocol": "Call", "version": "1.0.0"}
        )
        await transport.output().send_message(message)

        message = TransportMessageUrgentFrame(
            message={"event": "start", "streamSid": stream_sid, "start": {"streamSid": stream_sid}}
        )
        await transport.output().send_message(message)

    @audiobuffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels):
        await save_audio(client_name, audio, sample_rate, num_channels)

    async def end_call():
        await asyncio.sleep(duration_secs)
        await task.queue_frame(EndFrame())

    runner = PipelineRunner()

    await asyncio.gather(runner.run(task), end_call())


async def main():
    parser = argparse.ArgumentParser(description="Pipecat Twilio Chatbot Client")
    parser.add_argument("-u", "--url", type=str, required=True, help="specify the server URL")
    parser.add_argument(
        "-c", "--clients", type=int, required=True, help="number of concurrent clients"
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=DEFAULT_CLIENT_DURATION,
        help=f"duration of each client in seconds (default: {DEFAULT_CLIENT_DURATION})",
    )
    args, _ = parser.parse_known_args()

    clients = []
    for i in range(args.clients):
        clients.append(asyncio.create_task(run_client(f"client_{i}", args.url, args.duration)))
    await asyncio.gather(*clients)


if __name__ == "__main__":
    asyncio.run(main())
