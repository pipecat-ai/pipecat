#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Extend a turn-detecting STT's end-of-turn timing with a custom stop strategy.

Deepgram Flux detects turn boundaries server-side and proposes them to the
pipeline. The strategies decide what to do with those proposals, so a subclass
can shift the timing without giving up the service's detection.

``GracePeriodUserTurnStopStrategy`` below holds the turn open for a beat after
Flux proposes the end of it. If the user resumes ("...actually, make that
Tuesday") the pending stop is cancelled and the turn continues, so the
afterthought lands in the same user message instead of arriving after the bot
has already started answering.
"""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import Frame, LLMRunFrame, ProposedUserStartedSpeakingFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start import ExternalUserTurnStartStrategy
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

# How long to hold the turn open after the STT proposes the end of it.
GRACE_PERIOD_SECS = 1.2


class GracePeriodUserTurnStopStrategy(ExternalUserTurnStopStrategy):
    """Delay end-of-turn so a trailing afterthought can reopen the turn.

    The base strategy ends the turn as soon as the service's proposal resolves.
    This subclass schedules that finalization ``grace_period`` seconds out
    instead, and cancels it if the user starts speaking again — the afterthought
    then joins the same user message rather than arriving mid-response.

    The cost is added latency on every turn, so keep the grace period short.
    """

    def __init__(self, *, grace_period: float = GRACE_PERIOD_SECS, **kwargs):
        """Initialize the strategy.

        Args:
            grace_period: Seconds to hold the turn open after the service
                proposes the end of it.
            **kwargs: Additional arguments passed to the parent strategy.
        """
        super().__init__(**kwargs)
        self._grace_period = grace_period
        self._pending: asyncio.Task | None = None

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Cancel any pending finalization when the user starts speaking again."""
        if isinstance(frame, ProposedUserStartedSpeakingFrame):
            await self._cancel_pending(resumed=True)
        return await super().process_frame(frame)

    async def handle_user_turn_stopped(self):
        """Drop the pending finalization once the turn has ended."""
        await self._cancel_pending()
        await super().handle_user_turn_stopped()

    async def cleanup(self):
        """Clean up the strategy."""
        await self._cancel_pending()
        await super().cleanup()

    # The override point: the base strategy calls this to end the turn, so this
    # is where this subclass adjusts the timing.
    async def trigger_user_turn_stopped(self, *, enable_user_speaking_frames: bool | None = None):
        """Schedule the finalization instead of running it now."""
        if self._pending:
            return
        self._pending = self.create_task(
            self._finalize_after_grace_period(enable_user_speaking_frames)
        )

    async def _finalize_after_grace_period(self, enable_user_speaking_frames: bool | None):
        await asyncio.sleep(self._grace_period)
        self._pending = None
        logger.debug("Grace period elapsed with no new speech; ending the user turn")
        await super().trigger_user_turn_stopped(
            enable_user_speaking_frames=enable_user_speaking_frames
        )

    async def _cancel_pending(self, *, resumed: bool = False):
        """Drop the scheduled finalization.

        Args:
            resumed: Whether the user speaking again is what cancelled it, as
                opposed to the turn ending some other way or the strategy
                shutting down.
        """
        if not self._pending:
            return
        task, self._pending = self._pending, None
        await self.cancel_task(task)
        if resumed:
            logger.debug("User resumed speaking within the grace period; turn stays open")


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

    # Flux detects turns server-side and proposes them to the pipeline.
    stt = DeepgramFluxSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=(
                "You are a helpful assistant in a voice conversation. Your responses will be "
                "spoken aloud, so avoid emojis, bullet points, or other formatting that can't "
                "be spoken. Respond to what the user said in a creative, helpful, and brief way."
            ),
        ),
    )

    context = LLMContext()
    # Flux would normally recommend ExternalUserTurnStrategies, which resolves its
    # proposals immediately. Supplying our own keeps its turn detection driving the
    # conversation — the start strategy is the stock one — while swapping in the
    # stop strategy that adjusts when the turn ends.
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[ExternalUserTurnStartStrategy()],
                stop=[GracePeriodUserTurnStopStrategy()],
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation
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
