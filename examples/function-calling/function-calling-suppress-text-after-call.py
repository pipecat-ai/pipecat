#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Drop the text an LLM response streams after it decides to call a function.

Some models keep writing after a function call in the same response ("...your
order shipped yesterday"), guessing at a result they haven't seen. In a voice
bot that guess is spoken before the function even runs. The LLM service pushes
an ``LLMFunctionCallDetectedFrame`` as soon as a call appears in the stream, in
order with the response's text, so a processor between the LLM and the TTS can
drop what follows it. Text written before the call ("Let me check.") is kept.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseStartFrame,
    LLMFunctionCallDetectedFrame,
    LLMRunFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAIRealtimeSTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


class SuppressTextAfterFunctionCall(FrameProcessor):
    """Drop the LLM text that follows a function call in the same response."""

    def __init__(self, **kwargs):
        """Initialize the processor.

        Args:
            **kwargs: Additional arguments passed to the parent FrameProcessor.
        """
        super().__init__(**kwargs)
        self._suppressing = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Drop text frames between a function call and the end of its response."""
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._suppressing = False
        elif isinstance(frame, LLMFunctionCallDetectedFrame):
            logger.debug(f"{self}: function call detected, dropping the rest of the text")
            self._suppressing = True
        elif isinstance(frame, LLMTextFrame) and self._suppressing:
            logger.debug(f"{self}: dropped text after function call: {frame.text!r}")
            return

        await self.push_frame(frame, direction)


async def get_order_status(params: FunctionCallParams, order_id: str):
    """Look up the shipping status of an order.

    Args:
        order_id: The order number the user gave.
    """
    await params.result_callback({"order_id": order_id, "status": "out for delivery"})


# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "livekit": lambda: LiveKitParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    stt = OpenAIRealtimeSTTService(api_key=os.environ["OPENAI_API_KEY"])

    tts = OpenAITTSService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAITTSService.Settings(voice="ballad"),
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=(
                "You are a helpful order-support assistant in a voice conversation. Your "
                "responses will be spoken aloud, so avoid emojis, bullet points, or other "
                "formatting that can't be spoken. When the user asks about an order, say a "
                "short sentence such as 'Let me check that' and call get_order_status in the "
                "same reply."
            ),
        ),
    )

    context = LLMContext(tools=[get_order_status])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            SuppressTextAfterFunctionCall(),
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation.
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
