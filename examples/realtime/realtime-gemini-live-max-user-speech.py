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
  prompt and pushes a ``UserStoppedSpeakingFrame`` *upstream*, to the user
  aggregator. The aggregator's ``ForcedUserTurnStopStrategy`` picks it up,
  closes the turn properly (resetting the turn state and stop strategies), and
  broadcasts the real ``UserStoppedSpeakingFrame`` back downstream; the service
  maps that to ``activity_end`` and the model answers what it heard so far.

  Direction matters here. An earlier version of this example pushed the forced
  stop *downstream*, straight at the LLM. That ended Gemini's turn but left the
  aggregator's turn open, so the user's NEXT utterance never opened a new turn
  (no ``activity_start``) and the model ignored it until the turn-stop
  strategies eventually timed out. Closing the turn at the aggregator keeps the
  whole pipeline in sync.

  The timer intentionally does NOT reset on a mid-turn ``UserStoppedSpeaking`` /
  ``UserStartedSpeaking`` pair (a turn detector calling an early stop while the
  user rambles on). It stays armed until the bot actually takes the turn
  (``BotStartedSpeakingFrame``), so pauses can't be used to dodge the cap. If
  the timer fires while no user turn is open, it does nothing.

- ``MuteUntilBotDoneUserMuteStrategy`` closes the gap right after a forced
  cutoff. The user is usually still talking when the cap fires, and that
  trailing speech can take the turn straight back before the bot responds. The
  controller fires ``on_max_speech_forced`` when it cuts the turn; that *arms*
  the mute, which engages when the forced ``UserStoppedSpeakingFrame`` reaches
  the aggregator and holds until the next ``BotStoppedSpeakingFrame``, so the
  bot gets to speak. Arming (instead of muting immediately) matters: a muted
  aggregator drops incoming ``UserStoppedSpeakingFrame``, so muting the moment
  the cap trips could swallow the very frame that closes the turn.

Caveat: the interjection prompt is sent as realtime text inside the still-open
activity window, just before it closes. Gemini Live incorporating mid-turn text
alongside audio in a single turn is best-effort, so treat the interjection as a
steer, not a guarantee. The forced turn end (the core behavior) is reliable.

Caveat: muting drops the user's ``TranscriptionFrame`` too, so the cut-off tail
of a force-ended turn is not written to the LLM context. To still capture the
user's words (for logging or records), ``UserTranscriptCapture`` snoops the
``TranscriptionFrame`` between the user aggregator and the LLM, where it passes
on its way upstream before the aggregator consumes or mutes it. That recovers
the full transcript even when the mute drops it from context.
"""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputTextRawFrame,
    LLMRunFrame,
    TranscriptionFrame,
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
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.time import time_now_iso8601
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


class ForcedUserTurnStopStrategy(BaseUserTurnStopStrategy):
    """Close the user turn when a ``UserStoppedSpeakingFrame`` arrives at the aggregator.

    ``MaxUserSpeechController`` pushes its forced stop *upstream*; this strategy
    is what receives it inside the user aggregator and ends the turn properly
    (state reset, real ``UserStoppedSpeakingFrame`` broadcast, ``activity_end``
    at the service). The aggregator's own broadcasts never re-enter it, so the
    only ``UserStoppedSpeakingFrame`` this ever sees is a forced one.

    Why not the built-in ``ExternalUserTurnStopStrategy``? That strategy is made
    for pipelines where an external component drives BOTH turn edges: it never
    sees a turn start here, so its watchdog concludes the user is silent and
    closes every turn moments after it opens. It also suppresses the
    ``UserStoppedSpeakingFrame`` broadcast (its external emitter already sent
    one downstream), which in this topology is the frame Gemini needs.
    """

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Trigger the turn stop on a received (forced) ``UserStoppedSpeakingFrame``."""
        if isinstance(frame, UserStoppedSpeakingFrame):
            await self.trigger_user_turn_stopped()
        return ProcessFrameResult.CONTINUE


class MaxUserSpeechController(FrameProcessor):
    """Cap how long the user may speak, then hand the turn to the bot.

    Place this between the user aggregator and the LLM. It is an inverted
    ``UserIdleController``: instead of firing when the user stays *silent* too
    long, it fires when the user keeps *speaking* too long.

    Events:
        on_max_speech_forced(controller): Fired when the cap trips and the
            controller force-ends the user's turn. A companion user mute
            strategy can listen for this to mute the user until the bot is
            done, so the user can't immediately grab the turn back (see
            ``MuteUntilBotDoneUserMuteStrategy`` below).
    """

    def __init__(self, *, max_speech_secs: float = 8.0, interjection: str | None = None):
        """Initialize the controller.

        Args:
            max_speech_secs: How long the user may keep the turn before the bot
                takes over. Measured from the first turn start, and NOT reset by
                mid-turn stop/start pairs; only the bot speaking resets it.
            interjection: Optional steering prompt sent to the model right before
                the turn is force-ended. ``None`` just ends the turn.
        """
        super().__init__()
        self._max_speech_secs = max_speech_secs
        self._interjection = interjection
        self._timer: asyncio.Task | None = None
        self._user_turn_open = False
        self._register_event_handler("on_max_speech_forced")

    async def cleanup(self):
        """Cancel any armed timer so a pending fire can't push frames post-teardown."""
        await super().cleanup()
        await self._cancel_timer()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Track user speech and force the turn to end if it runs too long."""
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_turn_open = True
            # Arm the timer only if it isn't already running: a stop/start pair
            # inside what the user perceives as one turn (e.g. a turn detector
            # calling an early stop mid-ramble) must not reset the cap.
            if self._timer is None:
                self._start_timer()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # This is the aggregator's real turn stop (on a forced cutoff it is
            # the round trip of the stop we pushed upstream). Let it through so
            # the service sends its activity_end. The timer stays armed: if the
            # user grabs the turn back before the bot speaks, the cap still
            # applies to the combined turn.
            self._user_turn_open = False
        elif isinstance(frame, BotStartedSpeakingFrame):
            # The bot took the turn: the cap did its job (or wasn't needed).
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
            # Nobody is mid-turn (e.g. the bot never spoke after a normal stop,
            # so the timer was never cancelled). Forcing a stop now would send
            # the service an unpaired activity_end; just disarm.
            return
        logger.info(f"User spoke past {self._max_speech_secs}s; handing the turn to the bot")
        await self._call_event_handler("on_max_speech_forced")
        if self._interjection:
            await self.push_frame(InputTextRawFrame(text=self._interjection))
        # Send the stop UPSTREAM so the user aggregator (via its
        # ForcedUserTurnStopStrategy) closes the turn for the whole pipeline.
        # It then broadcasts the real UserStoppedSpeakingFrame downstream, which
        # passes back through here on its way to the service (activity_end).
        await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)


class MuteUntilBotDoneUserMuteStrategy(BaseUserMuteStrategy):
    """Mute the user from a forced cutoff until the bot finishes its turn.

    Without this, force-ending the user's turn has a gap: the user is often
    still talking when the cap fires, and that trailing speech can take the
    turn right back before the bot gets to respond. This strategy closes the
    gap. Call ``mute_until_bot_done()`` when the cap trips (wire it to
    ``MaxUserSpeechController``'s ``on_max_speech_forced`` event) and the user
    stays muted until the next ``BotStoppedSpeakingFrame``.

    The call *arms* the mute rather than engaging it: the mute engages when the
    forced ``UserStoppedSpeakingFrame`` reaches the aggregator, and applies from
    the frame after it. That ordering is deliberate. A muted aggregator drops
    incoming ``UserStoppedSpeakingFrame``, so muting immediately at cap time
    races the controller's upstream stop against the next audio frame; if audio
    wins, the mute swallows the stop and the turn never closes.

    While muted, the user aggregator drops the user's audio, VAD, and
    transcription frames, so the trailing speech can't reopen the turn. (One
    side effect: the dropped ``TranscriptionFrame`` means the cut-off tail of
    that turn won't be written to context.)
    """

    def __init__(self):
        """Initialize the strategy unmuted."""
        super().__init__()
        self._armed = False
        self._muted = False

    def mute_until_bot_done(self):
        """Arm the mute: it engages on the forced ``UserStoppedSpeakingFrame``."""
        self._armed = True

    async def process_frame(self, frame: Frame) -> bool:
        """Mute from the forced turn stop until the bot stops speaking.

        Args:
            frame: The frame to be processed.

        Returns:
            Whether the user should be muted right now.
        """
        await super().process_frame(frame)
        if self._armed and isinstance(frame, UserStoppedSpeakingFrame):
            # The forced stop reached the aggregator. The aggregator applies the
            # new mute state starting with the NEXT frame, so this stop still
            # closes the turn before the mute engages.
            self._armed = False
            self._muted = True
        elif self._muted and isinstance(frame, BotStoppedSpeakingFrame):
            self._muted = False
        return self._muted


class UserTranscriptCapture(FrameProcessor):
    """Capture the user's transcript even when the mute drops it from context.

    Place this between the user aggregator and the LLM. The realtime service's
    user ``TranscriptionFrame`` travels *upstream* (LLM -> aggregator), so it
    reaches this processor before the user aggregator can consume or mute it.
    That matters on a forced cutoff: ``MuteUntilBotDoneUserMuteStrategy`` mutes
    the user the instant the cap trips, and a muted aggregator drops
    ``TranscriptionFrame``, so the tail of the cut-off turn never lands in the
    LLM context. Snooping it here recovers the user's words for logging or
    records, independent of that muting.

    This captures the *text*; it does not put the dropped tail back into the LLM
    context for that turn. Replace the log call with whatever you need (append
    to a transcript store, emit an event, write to your DB).

    It also stamps when the user *started* talking. ``TranscriptionFrame.timestamp``
    is set by the STT/realtime service at the moment it pushes the frame, not when
    the user began speaking, so it lags the real turn start. We record the time as
    ``UserStartedSpeakingFrame`` passes (the same turn-start signal the cap keys
    off) and pair it with the transcript. If you want a more precise onset,
    ``VADUserStartedSpeakingFrame`` carries a wall-clock ``timestamp`` field you
    can read instead.
    """

    def __init__(self):
        """Initialize with no turn in progress."""
        super().__init__()
        self._turn_started_at: str | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Log every user transcript (with its turn-start time), then push it along."""
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            # Stamp the real speech start now, not the later push time that the
            # TranscriptionFrame carries.
            self._turn_started_at = time_now_iso8601()
        elif isinstance(frame, TranscriptionFrame) and frame.text.strip():
            logger.info(
                f"User transcript captured: {frame.text!r} "
                f"(turn started: {self._turn_started_at}, frame pushed: {frame.timestamp})"
            )
        await self.push_frame(frame, direction)


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
    # Mute the user from a forced cutoff until the bot is done, so trailing
    # speech can't grab the turn back. The controller's on_max_speech_forced
    # event (wired below) flips it on.
    mute_until_bot_done = MuteUntilBotDoneUserMuteStrategy()

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        realtime_service_mode=True,
        user_params=LLMUserAggregatorParams(
            # stop_secs is intentionally longer than Pipecat's 0.2s default:
            # manual-VAD mode does a bit better with end-of-speech padded with a
            # bit more silence.
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
            user_mute_strategies=[mute_until_bot_done],
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    # The default stop strategy, restated because listing any
                    # stop strategy replaces the defaults.
                    TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3()),
                    # Lets MaxUserSpeechController end the turn: it pushes a
                    # UserStoppedSpeakingFrame upstream and this strategy closes
                    # the turn at the aggregator when that frame arrives.
                    ForcedUserTurnStopStrategy(),
                ],
            ),
        ),
    )

    max_speech = MaxUserSpeechController(
        max_speech_secs=6.0,
        interjection=(
            "The user has been talking for a while. Politely interrupt and let "
            "them know you'd like to respond now."
        ),
    )

    @max_speech.event_handler("on_max_speech_forced")
    async def on_max_speech_forced(controller):
        # The cap tripped: mute the user until the bot finishes responding.
        mute_until_bot_done.mute_until_bot_done()

    # Sits between the user aggregator and the LLM so it sees the user's
    # upstream-traveling TranscriptionFrame before the (muted) aggregator drops
    # it. This is how we keep the cut-off turn's transcript despite the mute.
    transcript_capture = UserTranscriptCapture()

    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            transcript_capture,
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
