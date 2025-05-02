#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    LLMMessagesFrame,
    StartInterruptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection

load_dotenv(override=True)


class DebugObserver(BaseObserver):
    """Observer to log interruptions and bot speaking events to the console.

    Logs all frame instances of:
    - StartInterruptionFrame
    - BotStartedSpeakingFrame
    - BotStoppedSpeakingFrame

    This allows you to see the frame flow from processor to processor through the pipeline for these frames.
    Log format: [EVENT TYPE]: [source processor] → [destination processor] at [timestamp]s
    """

    async def on_push_frame(
        self,
        src: FrameProcessor,
        dst: FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
        timestamp: int,
    ):
        # Convert timestamp to seconds for readability
        time_sec = timestamp / 1_000_000_000

        # Create direction arrow
        arrow = "→" if direction == FrameDirection.DOWNSTREAM else "←"

        if isinstance(frame, StartInterruptionFrame):
            logger.info(f"⚡ INTERRUPTION START: {src} {arrow} {dst} at {time_sec:.2f}s")
        elif isinstance(frame, BotStartedSpeakingFrame):
            logger.info(f"🤖 BOT START SPEAKING: {src} {arrow} {dst} at {time_sec:.2f}s")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            logger.info(f"🤖 BOT STOP SPEAKING: {src} {arrow} {dst} at {time_sec:.2f}s")


@dataclass
class StartConversationFrame(Frame):
    """Frame to initiate a conversation.

    This frame is used to signal the start of a conversation in the pipeline.
    It can be used to trigger specific actions or responses from the system.
    """

    pass


class ConversationStarterProcessor(FrameProcessor):
    def __init__(self, message: str = "Hi! I'm a default message!"):
        super().__init__()
        self.message = message
        self._user_stopped_speaking_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Say a default message when the user starts speaking.

        This processor listens for the UserStartedSpeakingFrame and sends a default message
        when the user starts speaking for the first time.

        Args:
            frame: The frame to process
            direction: Direction of the frame flow
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, (StartConversationFrame, UserStoppedSpeakingFrame)):
            self._user_stopped_speaking_count += 1
            logger.info(
                f"++ {frame.name} User stopped speaking, count: {self._user_stopped_speaking_count}"
            )
            if self._user_stopped_speaking_count == 1:
                # First time user started speaking, send the message
                await self.push_frame(TTSSpeakFrame(self.message))
            else:
                await self.push_frame(frame)
        else:
            # Pass through other frames
            await self.push_frame(frame)


async def run_bot(webrtc_connection: SmallWebRTCConnection, _: argparse.Namespace):
    logger.info(f"Starting bot")

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    async def handle_user_idle(user_idle: UserIdleProcessor, retry_count: int) -> bool:
        logger.info(f"User idle, timeout : {user_idle._timeout} retry count: {retry_count}")
        if retry_count == 1:
            # First attempt: Trigger the conversation starter
            await user_idle.push_frame(StartConversationFrame())
            return True
        elif retry_count == 2:
            # Second attempt: More direct prompt
            messages.append(
                {
                    "role": "system",
                    "content": "The user is still inactive. Ask if they'd like to continue our conversation.",
                }
            )
            await user_idle.push_frame(LLMMessagesFrame(messages))
            return True
        else:
            # Third attempt: End the conversation
            await user_idle.push_frame(
                TTSSpeakFrame("It seems like you're busy right now. Have a nice day!")
            )
            await user_idle.push_frame(EndFrame(), FrameDirection.UPSTREAM)
            return False

    user_idle = UserIdleProcessor(callback=handle_user_idle, timeout=4.0)

    conversation_starter = ConversationStarterProcessor(message="This is a default message.")

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            user_idle,  # Idle user check-in
            conversation_starter,
            context_aggregator.user(),
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            report_only_initial_ttfb=True,
            observers=[DebugObserver()],
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Start the user idle timer
        await task.queue_frames([BotSpeakingFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")

    @transport.event_handler("on_client_closed")
    async def on_client_closed(transport, client):
        logger.info(f"Client closed connection")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


if __name__ == "__main__":
    from run import main

    main()
