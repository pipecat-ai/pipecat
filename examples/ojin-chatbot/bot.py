#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import io
import os
import sys
import tkinter as tk
from typing import Awaitable

import cv2
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from utils.tk_overlay import create_fps_overlay, start_tk_fps_udpater, start_tk_updater

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, ImageRawFrame, OutputImageRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.ojin.video import OjinPersonaService, OjinPersonaSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.transports.local.tk import TkLocalTransport, TkTransportParams

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class ImageFormatConverter(FrameProcessor):
    """Converts image frames from JPG/BGR format to PPM format."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, OutputImageRawFrame):
            try:
                # Get the image bytes data
                image_bytes = frame.image

                # Check if it's bytes data
                if isinstance(image_bytes, bytes):
                    # Decode the image from bytes (assuming it's JPEG format from Ojin)
                    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

                    # Decode using OpenCV
                    decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                    if decoded_image is None:
                        logger.error("Failed to decode image from bytes")
                        await self.push_frame(frame, direction)
                        return

                    # Convert from BGR to RGB (OpenCV loads as BGR by default)
                    rgb_image = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)

                    # Convert to raw RGB bytes (what Tkinter expects for PPM)
                    # The Tkinter transport will add the PPM header itself
                    rgb_bytes = rgb_image.tobytes()

                    # Create new frame with raw RGB data
                    converted_frame = OutputImageRawFrame(
                        image=rgb_bytes,  # Use raw RGB bytes
                        size=(rgb_image.shape[1], rgb_image.shape[0]),  # (width, height)
                        format="RGB",
                    )

                    await self.push_frame(converted_frame, direction)
                else:
                    await self.push_frame(frame, direction)

            except Exception as e:
                logger.error(f"Error converting image format: {e}")
                await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


async def main():
    tk_root = tk.Tk()
    tk_root.title("Ojin Persona Chatbot")

    # Configure window to be visible on Windows
    tk_root.geometry("1280x720")
    tk_root.resizable(True, True)
    tk_root.lift()
    tk_root.attributes("-topmost", True)
    tk_root.after_idle(tk_root.attributes, "-topmost", False)
    tk_root.focus_force()

    tk_transport = TkLocalTransport(
        tk_root,
        TkTransportParams(
            audio_in_enabled=False,
            audio_out_enabled=True,
            video_out_enabled=True,
            video_out_is_live=True,
            video_out_width=1280,
            video_out_height=720,
        ),
    )

    audio_transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_out_enabled=False,
            audio_in_enabled=True,
            vad_enabled=True,
            vad_audio_passthrough=True,
            vad_analyzer=SileroVADAnalyzer(),
        )
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID", ""),
    )

    # Initialize LLM service
    llm = HumeSTSService(
        api_key=self.config.get_variable(
            "HUME_API_KEY",
            "",
        ),
        config_id=self.config.get_variable(
            "HUME_CONFIG_ID",
            "",
        ),
        model=self.config.get_variable(
            "HUME_MODEL",
            "evi",
        ),
        system_prompt=self.config.welcome_settings.welcome_system_prompt,
    )

    messages = [
        {
            "role": "system",
            "content": "Always answer the last question from the context no matter who was the role of the last question",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # DITTO_SERVER_URL: str = "wss://eu-central-1.models.ojin.foo/realtime"
    persona = OjinPersonaService(
        OjinPersonaSettings(
            ws_url=os.getenv("OJIN_PROXY_URL", ""),
            api_key=os.getenv("OJIN_API_KEY", ""),
            persona_config_id=os.getenv("OJIN_PERSONA_ID", ""),
            image_size=(1280, 720),
            idle_to_speech_seconds=2.0,
            idle_sequence_duration=5,
            tts_audio_passthrough=False,
            push_bot_stopped_speaking_frames=False,
        )
    )

    # Create image format converter
    image_converter = ImageFormatConverter()

    # Create FPS overlay and start Tk updater
    fps_server_canvas = create_fps_overlay(tk_root, x=8, y=8, width=1280, height=120)
    fps_canvas = create_fps_overlay(tk_root, x=8, y=128, width=1280, height=120)
    tk_update_task = start_tk_updater(tk_root, interval_ms=10)
    tk_fps_update_task = start_tk_fps_udpater(
        tk_root, persona.fsm_fps_tracker, fps_canvas, interval_ms=80
    )
    tk_fps_server_update_task = start_tk_fps_udpater(
        tk_root, persona.server_fps_tracker, fps_server_canvas, interval_ms=80
    )

    pipeline = Pipeline(
        [
            audio_transport.input(),
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            persona,
            image_converter,  # Convert image format from BGR to PPM
            tk_transport.output(),  # Transport video output
            audio_transport.output(),  # Transport audio output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True, enable_usage_metrics=True, allow_interruptions=True
        ),
    )

    # messages.append({"role": "system", "content": "Please introduce yourself to the user."})
    # await task.queue_frames([context_aggregator.user().get_context_frame()])

    runner = PipelineRunner(handle_sigint=False)

    try:
        await runner.run(task)
    finally:
        # Clean up the Tkinter update task
        if "tk_update_task" in locals():
            tk_update_task.cancel()
            try:
                await tk_update_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    asyncio.run(main())
