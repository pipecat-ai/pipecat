#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import aiohttp
import os
import sys

from pipecat.frames.frames import Frame, LLMMessagesFrame, MetricsFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator, LLMUserResponseAggregator)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.chattts import ChatTTSTTSService
from pipecat.services.doubao import DoubaoLLMService
from pipecat.services.ollama import OLLamaLLMService
from pipecat.services.openai import OpenAILLMService
from pipecat.services.tencentstt import TencentSTTService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer

from runner import configure

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class MetricsLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, MetricsFrame):
            print(
                f"!!! MetricsFrame: {frame}, ttfb: {frame.ttfb}, processing: {frame.processing}, tokens: {frame.tokens}, characters: {frame.characters}")
        await self.push_frame(frame, direction)


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                audio_in_enabled=True,
                transcription_enabled=False,
                vad_enabled=True,
                vad_audio_passthrough=True,
                vad_analyzer=SileroVADAnalyzer(),
            )
        )
        stt = TencentSTTService()

        # you need to setup a ChatTTS service
        tts = ChatTTSTTSService(
            aiohttp_session=session,
            api_url=os.getenv("CHATTTS_API_URL", "http://localhost:8555/generate")
        )

        # llm = OLLamaLLMService()

        llm = DoubaoLLMService(
            model=os.getenv("DOUBAO_MODEL_ID"),  # DOUBAO_MODEL_ID
        )

        messages = [
            {
                "role": "system",
                "content": """你是一个智能客服, 友好的回答用户问题

                        注意：
                        用[uv_break]表示断句。
                        如果有需要笑的地方,请加[laugh],不要用[smile]。
                        如果有数字,输出中文数字,比如一,三十一,五百八十九。不要用阿拉伯数字。
                        """
            }
        ]

        ml = MetricsLogger()
        tma_in = LLMUserResponseAggregator(messages)
        tma_out = LLMAssistantResponseAggregator(messages)

        pipeline = Pipeline([
            transport.input(),   # Transport user input
            stt,                 # STT
            tma_in,              # User responses
            llm,                 # LLM
            tts,                 # TTS
            transport.output(),  # Transport bot output
            tma_out              # Assistant spoken responses
        ])

        task = PipelineTask(pipeline, PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            report_only_initial_ttfb=True,
        ))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])
        #     # Kick off the conversation.
            messages.append(
                {"role": "user", "content": "向用户问好.不超过10个字"})
            await task.queue_frames([LLMMessagesFrame(messages)])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
