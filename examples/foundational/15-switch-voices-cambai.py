#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSUpdateSettingsFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cambai.tts import CambAITTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
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
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


async def run_example(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
    logger.info(f"Starting CambAI voice switching bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CambAITTSService(
        api_key=os.getenv("CAMBAI_API_KEY"),
        voice_id=int(os.getenv("CAMBAI_VOICE_ID", "1")),
        sample_rate=24000,
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": """You are a helpful AI assistant that can demonstrate different CambAI voices. 
            
            When the user asks you to change your voice, respond with excitement about the change and then use the new voice. 
            
            Available voice commands:
            - "voice 1" or "voice one" - Switch to voice ID 1
            - "voice 2" or "voice two" - Switch to voice ID 2  
            - "voice 3" or "voice three" - Switch to voice ID 3
            - "original voice" or "default voice" - Switch back to voice ID 1
            
            When switching voices, acknowledge the change enthusiastically. Keep responses conversational and don't include special characters.""",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # Voice switching logic
    async def handle_voice_change(text: str):
        """Handle voice change commands"""
        text_lower = text.lower()
        
        if "voice 1" in text_lower or "voice one" in text_lower or "original voice" in text_lower or "default voice" in text_lower:
            await task.queue_frames([TTSUpdateSettingsFrame({"voice": "1"})])
            logger.info("Switched to voice 1")
        elif "voice 2" in text_lower or "voice two" in text_lower:
            await task.queue_frames([TTSUpdateSettingsFrame({"voice": "2"})])
            logger.info("Switched to voice 2")
        elif "voice 3" in text_lower or "voice three" in text_lower:
            await task.queue_frames([TTSUpdateSettingsFrame({"voice": "3"})])
            logger.info("Switched to voice 3")

    # Custom processor to detect voice change commands
    class VoiceChangeProcessor:
        async def process_frame(self, frame, direction):
            if hasattr(frame, 'text'):
                await handle_voice_change(frame.text)
            return frame

    voice_processor = VoiceChangeProcessor()

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # Speech-to-text
            voice_processor,  # Voice change detection
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # CambAI TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            report_only_initial_ttfb=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({
            "role": "system", 
            "content": """Please introduce yourself as a CambAI voice demonstration bot. 
            Explain that you can switch between different voices when asked. 
            Tell the user they can say things like 'switch to voice 2' or 'use voice 3' to hear different voices.
            Ask what they'd like to talk about or if they want to try different voices."""
        })
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


if __name__ == "__main__":
    from run import main

    main(run_example, transport_params=transport_params)
