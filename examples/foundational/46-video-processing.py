#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import os

import cv2
import numpy as np
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, InputImageRawFrame, LLMRunFrame, OutputImageRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.daily.transport import DailyParams, DailyTransport

load_dotenv(override=True)

transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_out_10ms_chunks=2,
        video_in_enabled=True,
        video_out_enabled=True,
        video_out_is_live=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_out_10ms_chunks=2,
        video_in_enabled=True,
        video_out_enabled=True,
        video_out_is_live=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


class EdgeDetectionProcessor(FrameProcessor):
    def __init__(self, video_out_width, video_out_height: int):
        super().__init__()
        self._video_out_width = video_out_width
        self._video_out_height = video_out_height

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Send back the user's camera video with edge detection applied
        if isinstance(frame, InputImageRawFrame) and frame.transport_source == "camera":
            # Convert bytes to NumPy array
            img = np.frombuffer(frame.image, dtype=np.uint8).reshape(
                (frame.size[1], frame.size[0], 3)
            )

            # perform edge detection only on camera frames
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            # convert the size if needed
            desired_size = (self._video_out_width, self._video_out_height)
            if frame.size != desired_size:
                resized_image = cv2.resize(img, desired_size)
                out_frame = OutputImageRawFrame(resized_image.tobytes(), desired_size, frame.format)
                await self.push_frame(out_frame)
            else:
                out_frame = OutputImageRawFrame(
                    image=img.tobytes(), size=frame.size, format=frame.format
                )
                await self.push_frame(out_frame)
        else:
            await self.push_frame(frame, direction)


SYSTEM_INSTRUCTION = f"""
"You are Gemini Chatbot, a friendly, helpful robot.

Your goal is to demonstrate your capabilities in a succinct way.

Your output will be converted to audio so don't include special characters in your answers.

Respond to what the user said in a creative and helpful way. Keep your responses brief. One or two sentences at most.
"""


async def run_bot(pipecat_transport):
    llm = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        voice_id="Puck",  # Aoede, Charon, Fenrir, Kore, Puck
        transcribe_user_audio=True,
        system_instruction=SYSTEM_INSTRUCTION,
    )

    messages = [
        {
            "role": "user",
            "content": "Start by greeting the user warmly and introducing yourself.",
        }
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # RTVI events for Pipecat client UI
    rtvi = RTVIProcessor()

    pipeline = Pipeline(
        [
            pipecat_transport.input(),
            context_aggregator.user(),
            rtvi,
            llm,  # LLM
            EdgeDetectionProcessor(
                pipecat_transport._params.video_out_width,
                pipecat_transport._params.video_out_height,
            ),  # Sending the video back to the user
            pipecat_transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("Pipecat client ready.")
        await rtvi.set_bot_ready()
        # Kick off the conversation.
        await task.queue_frames([LLMRunFrame()])

    @pipecat_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, participant):
        logger.info("Pipecat Client connected")
        if isinstance(transport, DailyTransport):
            await pipecat_transport.capture_participant_video(participant["id"], framerate=30)
        else:
            await pipecat_transport.capture_participant_video("camera")

    @pipecat_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Pipecat Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False, force_gc=True)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
