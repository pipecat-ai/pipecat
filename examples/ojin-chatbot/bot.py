#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
import io
from typing import Awaitable

from dotenv import load_dotenv
from loguru import logger
import cv2
import numpy as np
from PIL import Image

from pipecat.frames.frames import Frame, ImageRawFrame, OutputImageRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.tk import TkLocalTransport, TkTransportParams
from pipecat.services.ojin.video import OjinAvatarService, OjinAvatarSettings
import tkinter as tk

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
                        format="RGB"
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
    tk_root.title("Ojin Avatar Chatbot")
    
    # Configure window to be visible on Windows
    tk_root.geometry("1280x720")
    tk_root.resizable(True, True)
    tk_root.lift()
    tk_root.attributes('-topmost', True)
    tk_root.after_idle(tk_root.attributes, '-topmost', False)
    tk_root.focus_force()
    
    # Make Tkinter responsive by processing events periodically
    async def update_tk_periodically():
        while True:
            try:
                tk_root.update_idletasks()
                tk_root.update()
                await asyncio.sleep(0.01)  # 10ms delay
            except tk.TclError:
                break  # Window was closed
            except Exception as e:
                logger.error(f"Error updating Tkinter: {e}")
                break
    
    # Start the periodic updater as a background task
    tk_update_task = asyncio.create_task(update_tk_periodically())

    tk_transport = TkLocalTransport(
        tk_root,
        TkTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=True,
            video_out_is_live=True,
            video_out_width=1280,
            video_out_height=720,
        ),
    )

    #audio_in_transport = LocalAudioInputTransport(self._pyaudio, self._params)

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID", ""),
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # DITTO_SERVER_URL: str = "wss://eu-central-1.models.ojin.foo/realtime"
    avatar = OjinAvatarService(OjinAvatarSettings(
        ws_url=os.getenv("OJIN_PROXY_URL", ""),
        api_key=os.getenv("OJIN_API_KEY", ""),
        avatar_config_id=os.getenv("OJIN_AVATAR_ID", ""),        
        image_size=(1280, 720),
        idle_to_speech_seconds=1.5,
        idle_sequence_duration=30,
        tts_audio_passthrough=False,
    ))    

    # Create image format converter
    image_converter = ImageFormatConverter()
    
    pipeline = Pipeline(
        [
            tk_transport.input(),  # Transport user input
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            avatar,
            image_converter,  # Convert image format from BGR to PPM
            tk_transport.output(),  # Transport bot output
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

    # messages.append({"role": "system", "content": "Please introduce yourself to the user."})
    # await task.queue_frames([context_aggregator.user().get_context_frame()])

    runner = PipelineRunner(handle_sigint=False)

    try:
        await runner.run(task)
    finally:
        # Clean up the Tkinter update task
        if 'tk_update_task' in locals():
            tk_update_task.cancel()
            try:
                await tk_update_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":    
    asyncio.run(main())
