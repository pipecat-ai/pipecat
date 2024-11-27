#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
import time

import aiohttp
from loguru import logger
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (BotSpeakingFrame, EndFrame, Frame,
                                   InputAudioRawFrame, StartInterruptionFrame,
                                   StopInterruptionFrame, TextFrame,
                                   TranscriptionFrame, TTSAudioRawFrame,
                                   UserStartedSpeakingFrame,
                                   UserStoppedSpeakingFrame)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class DebugProcessor(FrameProcessor):
    def __init__(self, name, **kwargs):
        self._name = name
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not (
            isinstance(frame, InputAudioRawFrame)
            or isinstance(frame, BotSpeakingFrame)
            or isinstance(frame, UserStoppedSpeakingFrame)
            or isinstance(frame, TTSAudioRawFrame)
            or isinstance(frame, TextFrame)
        ):
            logger.debug(f"--- {self._name}: {frame} {direction}")
        await self.push_frame(frame, direction)


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport = DailyTransport(
            room_url, None, "AI Bot", DailyParams(
                audio_out_enabled=True,                 
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),)
        )
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
        ]

        dp = DebugProcessor("dp")

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        runner = PipelineRunner()

        task = PipelineTask(
            Pipeline(
                [
                    # transport.input(),
                    context_aggregator.user(),
                    dp,
                    llm,
                    tts,
                    transport.output(),
                    context_aggregator.assistant(),
                ]
            ),
            PipelineParams(
                allow_interruptions=True,
            ),
        )

        # Register an event handler so we can play the audio when the
        # participant joins.
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            participant_id = participant.get("info", {}).get("participantId", "")

            # Create frames for 60 seconds
            start_time = time.time()
            while time.time() - start_time < 300:
                elapsed_time = round(time.time() - start_time)
                logger.info(f"Running for {elapsed_time} seconds")
                await task.queue_frame(
                    StartInterruptionFrame(),
                )
                await asyncio.sleep(1) 
                
                await task.queue_frame(
                    UserStartedSpeakingFrame(),
                )
                
                await asyncio.sleep(1) 
                
                await task.queue_frame(
                    TranscriptionFrame("Tell a joke about dogs.", participant_id, time.time()),
                )

                await asyncio.sleep(1) 
                
                await task.queue_frame(
                    StopInterruptionFrame(),
                )

                await asyncio.sleep(1) 

                
                await task.queue_frame(
                    UserStoppedSpeakingFrame(),
                )
                
                await asyncio.sleep(5) 
                
                await task.queue_frame(
                        StartInterruptionFrame()
                )
                await asyncio.sleep(1) 
                
                await task.queue_frame(
                    UserStartedSpeakingFrame(),
                )
                
                await asyncio.sleep(1) 
                
                await task.queue_frame(
                    TranscriptionFrame("Tell a joke about cats.", participant_id, time.time()),
                )

                await asyncio.sleep(1) 
                
                await task.queue_frames(
                    StopInterruptionFrame(),
                )

                await asyncio.sleep(1) 
                await task.queue_frame(
                    UserStoppedSpeakingFrame(),
                )
                await asyncio.sleep(5)
            await task.queue_frame(EndFrame())
        
        
        # @transport.event_handler("on_first_participant_joined")
        # async def on_first_participant_joined(transport, participant):
        #     await transport.capture_participant_transcription(participant["id"])
        #     # Kick off the conversation.
        #     messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        #     await task.queue_frames([LLMMessagesFrame(messages)])

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
