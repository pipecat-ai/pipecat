#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import aiohttp
import os
import sys
import re
import time
from enum import Enum

from pipecat.frames.frames import (
    Frame, SystemFrame, TranscriptionFrame, TextFrame
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator, LLMUserResponseAggregator)
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer

from runner import configure

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class WakeCheckFilter(FrameProcessor):
    """
    This filter looks for wake phrases in the transcription frames and only passes through frames
    after a wake phrase has been detected. It also has a keepalive timeout to allow for a brief
    period of continued conversation after a wake phrase has been detected.
    """
    class WakeState(Enum):
        IDLE = 1
        AWAKE = 2

    class ParticipantState:
        def __init__(self, participant_id: str):
            self.participant_id = participant_id
            self.state = WakeCheckFilter.WakeState.IDLE
            self.wake_timer = 0
            self.accumulator = ""

    def __init__(self, wake_phrases: list[str], keepalive_timeout: float = 2):
        super().__init__()
        self._participant_states = {}
        self._keepalive_timeout = keepalive_timeout
        self._wake_patterns = []
        for name in wake_phrases:
            pattern = re.compile(r'\b' + r'\s*'.join(re.escape(word)
                                 for word in name.split()) + r'\b', re.IGNORECASE)
            self._wake_patterns.append(pattern)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        try:
            if isinstance(frame, TranscriptionFrame):
                p = self._participant_states.get(frame.user_id)
                if p is None:
                    p = WakeCheckFilter.ParticipantState(frame.user_id)
                    self._participant_states[frame.user_id] = p

                # If we have been AWAKE within the last keepalive_timeout seconds, pass
                # the frame through
                if p.state == WakeCheckFilter.WakeState.AWAKE:
                    if time.time() - p.wake_timer < self._keepalive_timeout:
                        logger.debug(
                            f"Wake phrase keepalive timeout has not expired. Passing frame through.")
                        p.wake_timer = time.time()
                        await self.push_frame(frame)
                        return
                    else:
                        p.state = WakeCheckFilter.WakeState.IDLE

                p.accumulator += frame.text
                for pattern in self._wake_patterns:
                    match = pattern.search(p.accumulator)
                    if match:
                        logger.debug(f"Wake phrase triggered: {match.group()}")
                        # Found the wake word. Discard from the accumulator up to the start of the match
                        # and modify the frame in place.
                        p.state = WakeCheckFilter.WakeState.AWAKE
                        p.wake_timer = time.time()
                        frame.text = p.accumulator[match.start():]
                        p.accumulator = ""
                        await self.push_frame(frame)
                    else:
                        pass
            else:
                await self.push_frame(frame, direction)
        except Exception as e:
            logger.error(f"Error in wake word filter: {e}")


async def main(room_url: str, token):

    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Robot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer()
            )
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Respond to what the user said in a creative and helpful way. Keep your responses brief.",
            },
        ]

        hey_robot_filter = WakeCheckFilter(["hey robot", "hey, robot"])
        tma_in = LLMUserResponseAggregator(messages)
        tma_out = LLMAssistantResponseAggregator(messages)

        pipeline = Pipeline([
            transport.input(),   # Transport user input
            hey_robot_filter,    # Filter out speech not directed at the robot
            tma_in,              # User responses
            llm,                 # LLM
            tts,                 # TTS
            transport.output(),  # Transport bot output
            tma_out              # Assistant spoken responses
        ])

        task = PipelineTask(pipeline, allow_interruptions=True)

        @ transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])
            await tts.say("Hi! If you want to talk to me, just say 'Hey Robot'.")

            # Kick off the conversation.
            # messages.append(
            #    {"role": "system", "content": "Please introduce yourself to the user."})
            # await task.queue_frames([LLMMessagesFrame(messages)])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
