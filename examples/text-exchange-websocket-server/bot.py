#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import BotInterruptionFrame, EndFrame, Frame, TextFrame, TranscriptionFrame, BotStoppedSpeakingFrame, StartInterruptionFrame, StopInterruptionFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame, LLMFullResponseEndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAIContextAggregatorPair, OpenAILLMService
from pipecat.transports.network.websocket_server import (
    WebsocketServerParams,
    WebsocketServerTransport,
    WebsocketServerOutputTransport,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class SessionTimeoutHandler:
    """Handles actions to be performed when a session times out.
    Inputs:
    - task: Pipeline task (used to queue frames).
    - tts: TTS service (used to generate speech output).
    """

    def __init__(self, task, tts):
        self.task = task
        self.tts = tts
        self.background_tasks = set()

    async def handle_timeout(self, client_address):
        """Handles the timeout event for a session."""
        try:
            logger.info(f"Connection timeout for {client_address}")

            # Queue a BotInterruptionFrame to notify the user
            await self.task.queue_frames([BotInterruptionFrame()])

            # Send the TTS message to inform the user about the timeout
            await self.tts.say(
                "I'm sorry, we are ending the call now. Please feel free to reach out again if you need assistance."
            )

            # Start the process to gracefully end the call in the background
            end_call_task = asyncio.create_task(self._end_call())
            self.background_tasks.add(end_call_task)
            end_call_task.add_done_callback(self.background_tasks.discard)
        except Exception as e:
            logger.error(f"Error during session timeout handling: {e}")

    async def _end_call(self):
        """Completes the session termination process after the TTS message."""
        try:
            # Wait for a duration to ensure TTS has completed
            await asyncio.sleep(15)

            # Queue both BotInterruptionFrame and EndFrame to conclude the session
            await self.task.queue_frames([BotInterruptionFrame(), EndFrame()])

            logger.info("TTS completed and EndFrame pushed successfully.")
        except Exception as e:
            logger.error(f"Error during call termination: {e}")


class ChatWebsocketServerOutputTransport(WebsocketServerOutputTransport):
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._write_frame(frame)
            self._next_send_time = 0
            return

        await super().process_frame(frame, direction)


class ChatWebsocketServerTransport(WebsocketServerTransport):
    def output(self) -> WebsocketServerOutputTransport:
        if not self._output:
            self._output = ChatWebsocketServerOutputTransport(self._params, name=self._output_name)
        return self._output


class InputTextFrameProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # In case we received a text sent as a transcription frame, we mock the interruption and speaking frames
        if isinstance(frame, TranscriptionFrame):
            await self.push_frame(StartInterruptionFrame())
            await self.push_frame(StopInterruptionFrame())
            await self.push_frame(UserStartedSpeakingFrame())

        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        # In case we received a text sent as a transcription frame, we mock the stopped speakig frames so the agent knows it can reply
        if isinstance(frame, TranscriptionFrame):
            await self.push_frame(UserStoppedSpeakingFrame())


class OutputTextFrameProcessor(FrameProcessor):
    _current_agent_transcript = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, TextFrame):
            self._current_agent_transcript = f"{self._current_agent_transcript} {frame.text}"

        if isinstance(frame, LLMFullResponseEndFrame) and direction == FrameDirection.DOWNSTREAM:
            text_frame = TextFrame(text=self._current_agent_transcript)
            await self.push_frame(text_frame, FrameDirection.DOWNSTREAM)
            self._current_agent_transcript = ""

        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


async def main():
    transport = ChatWebsocketServerTransport(
        params=WebsocketServerParams(
            audio_out_sample_rate=16000,
            audio_out_enabled=True,
            add_wav_header=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            session_timeout=60 * 3,  # 3 minutes
        )
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        sample_rate=16000,
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    input_text_frame_processor = InputTextFrameProcessor()
    output_text_frame_processor = OutputTextFrameProcessor()

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            stt,  # Speech-To-Text
            input_text_frame_processor,
            context_aggregator.user(),
            llm,  # LLM
            tts,  # Text-To-Speech
            output_text_frame_processor,
            transport.output(),  # Websocket output to client
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Kick off the conversation.
        messages.append({"role": "system", "content": "You are a helpful assistant."})
        # await task.queue_frames([context_aggregator.user().get_context_frame()])
        await tts.say("Hi, how can I help you?")

    @transport.event_handler("on_session_timeout")
    async def on_session_timeout(transport, client):
        logger.info(f"Entering in timeout for {client.remote_address}")

        timeout_handler = SessionTimeoutHandler(task, tts)

        await timeout_handler.handle_timeout(client)

    runner = PipelineRunner()

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
