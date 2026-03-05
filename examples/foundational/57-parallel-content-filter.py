#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    ControlFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMRunFrame,
    SystemFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


@dataclass
class ContentApprovedFrame(ControlFrame):
    """Signal frame indicating content passed the filter."""

    pass


@dataclass
class ContentRejectedFrame(ControlFrame):
    """Signal frame indicating content was rejected by the filter."""

    pass


FILTERED_WORDS = ["apple", "banana", "car"]


class ContentFilterProcessor(FrameProcessor):
    """Checks LLMContextFrames for filtered words and emits signal frames.

    Runs in one branch of a ParallelPipeline. Emits ContentApprovedFrame or
    ContentRejectedFrame so that a downstream ContentFilterGate can decide
    whether to let the LLM's output through.
    """

    def _contains_filtered_words(self, context: "LLMContext") -> bool:
        """Check if the last message in the context contains any filtered words."""
        messages = context.messages
        if messages:
            last_message = messages[-1]
            content = last_message.get("content", "") if isinstance(last_message, dict) else ""
            if isinstance(content, str):
                content_lower = content.lower()
                if any(word in content_lower for word in FILTERED_WORDS):
                    logger.info(f"Filtered content detected: {content}")
                    return True
        return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            if self._contains_filtered_words(frame.context):
                await self.push_frame(ContentRejectedFrame(), direction)
            else:
                await self.push_frame(ContentApprovedFrame(), direction)
            return

        await self.push_frame(frame, direction)


class ContentFilterGate(FrameProcessor):
    """Gates LLM output until the content filter signals approval or rejection.

    Placed after a ParallelPipeline that runs a ContentFilterProcessor alongside
    an LLM. Because the content filter (a fast regex check) completes before the
    LLM's first token arrives, the signal frame always reaches this gate first.

    - On ContentApprovedFrame: subsequent LLM output passes through normally.
    - On ContentRejectedFrame: LLM output is discarded and a canned rejection
      message is spoken instead via TTSSpeakFrame.

    Note: For a production implementation with a slow content filter (e.g. an
    external moderation API), you would add frame buffering so that LLM output
    arriving before the filter decision is held rather than passed through.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rejecting = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # System frames always pass through.
        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
            return

        # Content filter approved — LLM output will pass through normally.
        if isinstance(frame, ContentApprovedFrame):
            return

        # Content filter rejected — suppress LLM output and speak a rejection.
        if isinstance(frame, ContentRejectedFrame):
            self._rejecting = True
            await self.push_frame(
                TTSSpeakFrame(text="I'm sorry, I can't respond to that."), direction
            )
            return

        # LLMFullResponseEndFrame marks the end of the LLM's response.
        # When rejecting, consume it to finish suppression.
        if isinstance(frame, LLMFullResponseEndFrame) and self._rejecting:
            self._rejecting = False
            return

        # While rejecting, discard all other frames (LLM text, etc.).
        if self._rejecting:
            return

        await self.push_frame(frame, direction)


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY", ""))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)
    content_filter = ContentFilterProcessor()
    content_gate = ContentFilterGate()

    # The content filter and LLM run in parallel. The content filter emits
    # a signal frame (approved/rejected) while the LLM generates text
    # concurrently. The gate after the ParallelPipeline blocks output until
    # the content filter decides. TTS is placed after the gate so rejected
    # content never reaches it.
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            context_aggregator.user(),  # User responses
            ParallelPipeline(
                [content_filter],  # Branch 1: content filter (emits signal frames)
                [llm],  # Branch 2: LLM text generation
            ),
            content_gate,  # Gates output until content filter approves
            tts,  # TTS (only processes approved text)
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
