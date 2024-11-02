#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import io
import os
import sys
from collections import deque

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, ImageRawFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.anthropic import AnthropicLLMContext, AnthropicLLMService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

MAX_FRAMES = 5
FRAMES_PER_SECOND = 0.2


video_participant_id = None

anthropic_context = None

recent_image_frames = deque(maxlen=MAX_FRAMES)


class ImageFrameCatcher(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        global recent_image_frames

        await super().process_frame(frame, direction)
        if isinstance(frame, ImageRawFrame):
            recent_image_frames.append(frame)
        else:
            await self.push_frame(frame, direction)


class TranscriptFrameCatcher(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            logger.debug(
                f"TranscriptLogger: {frame}, num frames: {len(recent_image_frames)}, anthropic context: {anthropic_context}"
            )
            if anthropic_context:
                add_message_with_images(
                    anthropic_context, frame.text, frames=list(recent_image_frames)
                )
        await self.push_frame(frame, direction)


async def main():
    global llm
    global anthropic_context

    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        llm = AnthropicLLMService(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-5-sonnet-20240620",
            enable_prompt_caching_beta=True,
        )

        # todo: test with very short initial user message

        system_prompt = """\
You are a helpful assistant who converses with a user and answers questions. Respond concisely to general questions. Keep
your answers brief unless explicitly asked for more information.

Your response will be turned into speech so use only simple words and punctuation.
        """

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                    }
                ],
            },
            {"role": "user", "content": "Start the conversation by saying 'hello'."},
        ]

        context = OpenAILLMContext(messages)
        anthropic_context = AnthropicLLMContext.upgrade_to_anthropic(context)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                ImageFrameCatcher(),
                TranscriptFrameCatcher(),
                context_aggregator.user(),  # User speech to text
                llm,  # LLM
                tts,  # TTS
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses and tool context
            ],
        )

        task = PipelineTask(
            pipeline, PipelineParams(allow_interruptions=False, enable_metrics=True)
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            global video_participant_id
            video_participant_id = participant["id"]
            await transport.capture_participant_transcription(video_participant_id)
            await transport.capture_participant_video(
                video_participant_id, framerate=FRAMES_PER_SECOND, video_source="screenVideo"
            )
            # Kick off the conversation.
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_app_message")
        async def on_app_message(transport, message, sender):
            logger.debug(f"Received app message: {message} - {context}")
            if not recent_image_frames:
                logger.debug("No image frames to send")
                return

            add_message_with_images(
                anthropic_context, message["message"], frames=list(recent_image_frames)
            )

            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()
        await runner.run(task)


def add_message_with_images(c, message, frames=None):
    if frames is None:
        frames = list(recent_image_frames)

    if not frames:
        logger.debug("No image frames to send")
        return

    # Create content list starting with all images
    content = []
    for frame in frames:
        buffer = io.BytesIO()
        Image.frombytes(frame.format, frame.size, frame.image).save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": encoded_image,
                },
            }
        )

    # Add text message at the end if provided
    if message:
        content.append({"type": "text", "text": message})

    logger.debug(f"Adding message: {content}")
    c.add_message({"role": "user", "content": content})


if __name__ == "__main__":
    asyncio.run(main())
