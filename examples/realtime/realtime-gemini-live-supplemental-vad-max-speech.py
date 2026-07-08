#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""NEGATIVE example: a max-user-speech cap with Gemini Live driving the turns.

This bot tries to cap how long a single user turn can run while Gemini Live's
server-side VAD drives the conversation. It exists to document (and to pin
down in an eval) that this combination does NOT work: with server VAD on, the
service never sends explicit activity signals, so the forced turn stop closes
the turn locally but Gemini keeps listening and only responds on its own turn
decision. See `GeminiLiveLLMService` in
`src/pipecat/services/google/gemini_live/llm.py`: `activity_start` /
`activity_end` are only sent when `vad=GeminiVADParams(disabled=True)`.

The `gemini_vad_max_speech_cap` release eval asserts the current behavior
(no inference starts inside the cap window), so it passes today. If that eval
ever FAILS, the service gained a turn-end lever in server-VAD mode and this
file, the Gemini Live docs Notes section, and the locally-driven cap example
should be revisited.

For the cap done right (server VAD off, local turns driving), see the
`jh/gemini-live-max-user-speech-example` branch.
"""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    Frame,
    LLMRunFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnMessageAddedMessage,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

system_instruction = """
You are a helpful assistant on a voice call. Keep your answers short: one or
two sentences. If you are ever told the user has been talking for a while,
politely jump in and take your turn.
"""


class ForcedUserTurnStopStrategy(BaseUserTurnStopStrategy):
    """Close the user turn when a ``UserStoppedSpeakingFrame`` arrives at the aggregator.

    Same shape as in ``realtime-gemini-live-max-user-speech.py``: the cap
    controller pushes its forced stop upstream and this strategy ends the turn
    at the aggregator. The difference here is what happens next: with server
    VAD on, the service does NOT translate the closed turn into an
    ``activity_end``, which is the point this example demonstrates.
    """

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Trigger the turn stop on a received (forced) ``UserStoppedSpeakingFrame``."""
        if isinstance(frame, UserStoppedSpeakingFrame):
            await self.trigger_user_turn_stopped()
        return ProcessFrameResult.CONTINUE


class MaxUserSpeechController(FrameProcessor):
    """Cap how long the user may speak, then try to hand the turn to the bot.

    Trimmed copy of the controller from
    ``realtime-gemini-live-max-user-speech.py`` (no interjection, no mute
    wiring): it starts a timer on the first ``UserStartedSpeakingFrame`` of a
    turn and, if the user is still going at ``max_speech_secs``, pushes a
    ``UserStoppedSpeakingFrame`` upstream to force the turn closed.
    """

    def __init__(self, *, max_speech_secs: float = 6.0):
        super().__init__()
        self._max_speech_secs = max_speech_secs
        self._timer: asyncio.Task | None = None
        self._user_turn_open = False

    async def cleanup(self):
        """Cancel any armed timer so a pending fire can't push frames post-teardown."""
        await super().cleanup()
        await self._cancel_timer()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Track user speech and force the turn to end if it runs too long."""
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_turn_open = True
            # Don't reset on mid-turn stop/start pairs; only the bot speaking
            # resets the cap.
            if self._timer is None:
                self._start_timer()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_turn_open = False
        elif isinstance(frame, BotStartedSpeakingFrame):
            await self._cancel_timer()
        await self.push_frame(frame, direction)

    def _start_timer(self):
        self._timer = self.create_task(self._expire())

    async def _cancel_timer(self):
        if self._timer:
            await self.cancel_task(self._timer)
            self._timer = None

    async def _expire(self):
        await asyncio.sleep(self._max_speech_secs)
        self._timer = None
        if not self._user_turn_open:
            return
        logger.info(f"User spoke past {self._max_speech_secs}s; forcing the turn to end")
        await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)


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
    logger.info(f"Starting bot")

    llm = GeminiLiveLLMService(
        api_key=os.environ["GOOGLE_API_KEY"],
        settings=GeminiLiveLLMService.Settings(
            system_instruction=system_instruction,
            voice="Aoede",
            # Server VAD stays ON: Gemini drives the turns. That is the whole
            # experiment; see the module docstring.
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # Supplemental local VAD: gives the cap controller its turn-start
            # signal (and RTVI its turn frames).
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=UserTurnStrategies(
                stop=[ForcedUserTurnStopStrategy()],
            ),
        ),
    )

    max_speech = MaxUserSpeechController(max_speech_secs=6.0)

    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            max_speech,
            llm,
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
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await worker.cancel()

    @user_aggregator.event_handler("on_user_turn_message_added")
    async def on_user_turn_message_added(aggregator, message: UserTurnMessageAddedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}user: {message.content}"
        logger.info(f"Transcript: {line}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}assistant: {message.content}"
        logger.info(f"Transcript: {line}")

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
