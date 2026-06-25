#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini Live with a maximum user speaking time.

Some bots need to cap how long a single user turn can run: a user who rambles
forever never hits Gemini's silence-based turn end, so the bot waits and waits.
This example caps the user's turn and hands it to the bot.

How it works:

- Gemini Live's server-side VAD is disabled (``GeminiVADParams(disabled=True)``)
  and a local ``SileroVADAnalyzer`` drives turns instead. In this mode the
  service maps Pipecat's ``UserStartedSpeakingFrame`` /
  ``UserStoppedSpeakingFrame`` to Gemini ``activity_start`` / ``activity_end``
  signals. That mapping is the lever: a ``UserStoppedSpeakingFrame`` ends the
  user's turn at the API and lets the model respond. With Gemini's own
  server-side VAD on, there is no such lever, so this pattern requires local
  turns.

- ``MaxUserSpeechController`` sits between the user aggregator and the LLM. It
  starts a timer when the user starts speaking and, if they are still going when
  ``max_speech_secs`` elapses, it (optionally) injects a short interjection
  prompt and pushes a ``UserStoppedSpeakingFrame`` downstream. The service sends
  ``activity_end`` and the model answers what it heard so far.

Caveat: the interjection prompt is sent as realtime text inside the still-open
activity window, just before it closes. Gemini Live incorporating mid-turn text
alongside audio in a single turn is best-effort, so treat the interjection as a
steer, not a guarantee. The forced turn end (the core behavior) is reliable.
"""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import (
    Frame,
    InputTextRawFrame,
    LLMRunFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, GeminiVADParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


class MaxUserSpeechController(FrameProcessor):
    """Cap how long the user may speak, then hand the turn to the bot.

    Place this between the user aggregator and the LLM. It is an inverted
    ``UserIdleController``: instead of firing when the user stays *silent* too
    long, it fires when the user keeps *speaking* too long.
    """

    def __init__(self, *, max_speech_secs: float = 8.0, interjection: str | None = None):
        """Initialize the controller.

        Args:
            max_speech_secs: How long a single user turn may run before the bot
                takes over.
            interjection: Optional steering prompt sent to the model right before
                the turn is force-ended. ``None`` just ends the turn.
        """
        super().__init__()
        self._max_speech_secs = max_speech_secs
        self._interjection = interjection
        self._timer: asyncio.Task | None = None
        self._forced = False

    async def cleanup(self):
        """Cancel any armed timer so a pending fire can't push frames post-teardown."""
        await super().cleanup()
        await self._cancel_timer()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Track user speech and force the turn to end if it runs too long."""
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._forced = False
            await self._start_timer()
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._cancel_timer()
            if self._forced:
                # We already ended this turn ourselves. Drop the aggregator's
                # real stop so the LLM sees a single, balanced start/stop pair
                # (and the service doesn't send a second activity_end).
                self._forced = False
                return
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _start_timer(self):
        await self._cancel_timer()
        self._timer = self.create_task(self._expire())

    async def _cancel_timer(self):
        if self._timer:
            await self.cancel_task(self._timer)
            self._timer = None

    async def _expire(self):
        await asyncio.sleep(self._max_speech_secs)
        self._timer = None
        logger.info(f"User spoke past {self._max_speech_secs}s; handing the turn to the bot")
        self._forced = True
        if self._interjection:
            await self.push_frame(InputTextRawFrame(text=self._interjection))
        await self.push_frame(UserStoppedSpeakingFrame())


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
            voice="Aoede",  # Puck, Charon, Kore, Fenrir, Aoede
            # Disable Gemini's server-side VAD so local turns drive the
            # conversation and UserStoppedSpeakingFrame becomes our turn-end lever.
            vad=GeminiVADParams(disabled=True),
        ),
    )

    context = LLMContext(
        [
            {
                "role": "user",
                "content": (
                    "You are a friendly assistant on a voice call. Greet the user "
                    "in one short sentence and invite them to tell you what's on "
                    "their mind. If you are ever told the user has been talking for "
                    "a while, politely jump in and take your turn."
                ),
            },
        ],
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        realtime_service_mode=True,
        user_params=LLMUserAggregatorParams(
            # stop_secs is intentionally longer than Pipecat's 0.2s default:
            # manual-VAD mode does a bit better with end-of-speech padded with a
            # bit more silence.
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
        ),
    )

    max_speech = MaxUserSpeechController(
        max_speech_secs=6.0,
        interjection=(
            "The user has been talking for a while. Politely interrupt and let "
            "them know you'd like to respond now."
        ),
    )

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
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await worker.cancel()

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
