#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import datetime
import io
import json
import os
import sys
import wave

import aiofiles
from dotenv import load_dotenv
from fastapi import WebSocket
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.atoms.atoms_agent import (
    AtomsAgentContext,
    CallData,
    initialize_conversational_agent,
)
from pipecat.services.atoms.prompts import FT_RESPONSE_MODEL_SYSTEM_PROMPT
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.waves.tts import WavesHttpTTSService, WavesSSETTSService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def save_audio(server_name: str, audio: bytes, sample_rate: int, num_channels: int):
    if len(audio) > 0:
        filename = (
            f"{server_name}_recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
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


async def run_bot(websocket_client: WebSocket, stream_sid: str, call_sid: str, testing: bool):
    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        audio_passthrough=True,
    )

    # tts = CartesiaTTSService(
    #     api_key=os.getenv("CARTESIA_API_KEY"),
    #     voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    #     push_silence_after_stop=testing,
    # )

    # tts = WavesHttpTTSService(
    #     api_key=os.getenv("WAVES_API_KEY"),
    #     voice_id="deepika",
    # )

    tts = WavesSSETTSService(
        api_key=os.getenv("WAVES_API_KEY"),
        voice_id="nyah",
    )

    agent = await initialize_conversational_agent(
        agent_id="680c74afa49c52c0f832821d",
        call_id="CALL-1746008459293-112543",
        call_data=CallData(
            variables={
                "call_id": "CALL-1747224604782-94a54d",
                "user_number": "+918815141212",
                "agent_number": "+918815141212",
            }
        ),
    )

    messages = [
        {
            "role": "system",
            "content": FT_RESPONSE_MODEL_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)
    initial_context = AtomsAgentContext.upgrade_to_atoms_agent(context)
    chunks = [chunk async for chunk in (await agent.get_response(context=initial_context))]
    initial_agent_response = "".join(chunks)

    logger.info(f"Initial agent response: {initial_agent_response}")

    # NOTE: Watch out! This will save all the conversation in memory. You can
    # pass `buffer_size` to get periodic callbacks.
    audiobuffer = AudioBufferProcessor(user_continuous_stream=not testing)

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            stt,  # Speech-To-Text
            context_aggregator.user(),
            # llm,  # LLM
            agent,
            tts,
            transport.output(),
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

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Start recording.
        await audiobuffer.start_recording()
        # Kick off the conversation.
        # await task.queue_frames([context_aggregator.user().get_context_frame()])
        await task.queue_frames([TTSSpeakFrame(text=initial_agent_response)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    @audiobuffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels):
        server_name = f"server_{websocket_client.client.port}"
        await save_audio(server_name, audio, sample_rate, num_channels)

    # We use `handle_sigint=False` because `uvicorn` is controlling keyboard
    # interruptions. We use `force_gc=True` to force garbage collection after
    # the runner finishes running a task which could be useful for long running
    # applications with multiple clients connecting.
    runner = PipelineRunner(handle_sigint=False, force_gc=True)

    await runner.run(task)
