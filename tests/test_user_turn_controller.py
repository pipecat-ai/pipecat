#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
import warnings
from unittest.mock import AsyncMock, MagicMock

from loguru import logger

from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    Frame,
    LLMContextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    UserTurnCompletionFrame,
    UserTurnInferenceCompletedFrame,
    UserTurnResponseCancelFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.turns.types import (
    ProcessFrameResult,
    UserTurnResumedParams,
    UserTurnSpeculation,
)
from pipecat.turns.user_start import (
    BaseUserTurnStartStrategy,
    ExternalUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
    UserTurnStartedParams,
    VADUserTurnStartStrategy,
    WakePhraseUserTurnStartStrategy,
)
from pipecat.turns.user_start.min_words_user_turn_start_strategy import (
    MinWordsUserTurnStartStrategy,
)
from pipecat.turns.user_stop import (
    BaseUserTurnStopStrategy,
    ExternalUserTurnCompletionStopStrategy,
    ExternalUserTurnStopStrategy,
    LLMTurnCompletionUserTurnStopStrategy,
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
    UserTurnStoppedParams,
    deferred,
)
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.turns.user_turn_strategies import (
    ExternalUserTurnStrategies,
    StrategyMode,
    UserTurnStrategies,
    UserTurnStrategiesParams,
)
from pipecat.utils import deprecation
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup

USER_TURN_STOP_TIMEOUT = 0.2
TRANSCRIPTION_TIMEOUT = 0.1


class TestUserTurnController(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.task_manager = TaskManager()

    async def test_completion_dropped_while_user_speaking(self):
        """A completion arriving while the user speaks must not stop the turn.

        External completions (e.g. an LLM ●) resolve with latency, so the user
        may have resumed speaking. The controller drops the finalization while
        the user is speaking and finalizes once they fall silent.
        """
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[ExternalUserTurnCompletionStopStrategy()],
            ),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )
        await controller.setup(frame_processor_setup(self.task_manager))
        await controller.start()

        stopped = False

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            nonlocal stopped
            stopped = True

        # Start a turn; the user is now speaking.
        await controller.process_frame(VADUserStartedSpeakingFrame())

        # A completion resolving while the user still speaks is stale: dropped.
        await controller.process_frame(UserTurnInferenceCompletedFrame())
        self.assertFalse(stopped)

        # Once the user stops, a fresh completion finalizes the turn.
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await controller.process_frame(UserTurnInferenceCompletedFrame())
        self.assertTrue(stopped)

        await controller.cleanup()

    async def test_default_user_turn_strategies(self):
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)],
            )
        )

        await controller.setup(frame_processor_setup(self.task_manager))
        await controller.start()

        should_start = None
        should_stop = None

        @controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            nonlocal should_start
            should_start = True

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            nonlocal should_stop
            should_stop = True

        await controller.process_frame(VADUserStartedSpeakingFrame())
        self.assertTrue(should_start)
        self.assertFalse(should_stop)

        await controller.process_frame(
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now")
        )
        self.assertTrue(should_start)
        self.assertFalse(should_stop)

        await controller.process_frame(VADUserStoppedSpeakingFrame())
        self.assertTrue(should_start)
        # Wait for user_speech_timeout to elapse
        await asyncio.sleep(TRANSCRIPTION_TIMEOUT + 0.1)
        self.assertTrue(should_stop)

    async def test_update_strategies_does_not_accumulate_event_handlers(self):
        # Regression: re-applying strategies (e.g. realtime mode re-installing the
        # same instances, or repeated metadata broadcasts) must not accumulate
        # duplicate controller event handlers on the strategy instances —
        # _cleanup_strategies removes what _setup_strategies added.
        start = VADUserTurnStartStrategy()
        stop = SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(start=[start], stop=[stop])
        )
        await controller.setup(frame_processor_setup(self.task_manager))
        await controller.start()

        # Re-apply the same strategy instances several times.
        for _ in range(3):
            await controller.update_strategies(UserTurnStrategies(start=[start], stop=[stop]))

        for strategy in (start, stop):
            for name, handler in strategy._event_handlers.items():
                self.assertLessEqual(
                    len(handler.handlers), 1, f"{name} double-registered on {strategy}"
                )

    async def test_inference_triggered_fires_alongside_stopped(self):
        """Default strategies fire both inference-triggered and stopped, in order."""
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)],
            )
        )

        await controller.setup(frame_processor_setup(self.task_manager))
        await controller.start()

        events: list[str] = []

        @controller.event_handler("on_user_turn_inference_triggered")
        async def on_user_turn_inference_triggered(controller, strategy, speculation):
            events.append("inference_triggered")

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            events.append("stopped")

        await controller.process_frame(VADUserStartedSpeakingFrame())
        await controller.process_frame(
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now")
        )
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(TRANSCRIPTION_TIMEOUT + 0.1)

        self.assertEqual(events, ["inference_triggered", "stopped"])

    async def test_deferred_wrapper_skips_stopped(self):
        """With the first decision winning, deferred() drops the inner strategy's stop."""
        wrapped = deferred(
            SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)
        )
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(stop=[wrapped]),
            user_turn_strategies_params=UserTurnStrategiesParams(stop_mode=StrategyMode.FIRST),
        )

        await controller.setup(frame_processor_setup(self.task_manager))
        await controller.start()

        events: list[str] = []

        @controller.event_handler("on_user_turn_inference_triggered")
        async def on_user_turn_inference_triggered(controller, strategy, speculation):
            events.append("inference_triggered")

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            events.append("stopped")

        await controller.process_frame(VADUserStartedSpeakingFrame())
        await controller.process_frame(
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now")
        )
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(TRANSCRIPTION_TIMEOUT + 0.1)

        # The inner strategy fires inference-triggered (forwarded by the
        # wrapper). Finalization is suppressed, but the controller's
        # stop watchdog eventually fires `stopped`.
        self.assertEqual(events[0], "inference_triggered")

    async def test_user_turn_start_reset(self):
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[MinWordsUserTurnStartStrategy(min_words=3)]
            ),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )

        await controller.setup(frame_processor_setup(self.task_manager))
        await controller.start()

        should_start = 0

        @controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            nonlocal should_start
            should_start += 1

        await controller.process_frame(BotStartedSpeakingFrame())
        await controller.process_frame(TranscriptionFrame(text="One", user_id="cat", timestamp=""))
        self.assertEqual(should_start, 0)

        await controller.process_frame(
            TranscriptionFrame(text="One two three!", user_id="cat", timestamp="")
        )
        self.assertEqual(should_start, 1)

        # Trigger user stop turn so we can trigger user start turn again.
        await asyncio.sleep(USER_TURN_STOP_TIMEOUT + 0.1)

        await controller.process_frame(BotStartedSpeakingFrame())
        await controller.process_frame(TranscriptionFrame(text="Hi!", user_id="cat", timestamp=""))
        self.assertEqual(should_start, 1)

        await controller.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertEqual(should_start, 2)

    async def test_user_turn_stop_timeout_no_transcription(self):
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )

        await controller.setup(frame_processor_setup(self.task_manager))
        await controller.start()

        should_start = None
        should_stop = None
        timeout = None

        @controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            nonlocal should_start
            should_start = True

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            nonlocal should_stop
            should_stop = True

        @controller.event_handler("on_user_turn_stop_timeout")
        async def on_user_turn_stop_timeout(controller):
            nonlocal timeout
            timeout = True

        await controller.process_frame(VADUserStartedSpeakingFrame())
        self.assertTrue(should_start)
        self.assertFalse(should_stop)
        self.assertFalse(timeout)

        await controller.process_frame(VADUserStoppedSpeakingFrame())
        self.assertTrue(should_start)
        self.assertFalse(should_stop)

        await asyncio.sleep(USER_TURN_STOP_TIMEOUT + 0.1)
        self.assertTrue(should_start)
        self.assertTrue(should_stop)
        self.assertTrue(timeout)

    async def test_a_turn_ended_by_the_timeout_keeps_a_confirmation(self):
        """The stop timeout ends the turn with what the decisions said, a confirmation included."""
        confirming, on_inference = DecidingStopStrategy(), DecidingStopStrategy(on_inference=True)
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(stop=[confirming, on_inference]),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )
        await controller.setup(frame_processor_setup(self.task_manager))
        await controller.start()
        stops = []

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            stops.append(params)

        await controller.process_frame(VADUserStartedSpeakingFrame())
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await confirming.speculate()
        await confirming.confirm()
        await asyncio.sleep(USER_TURN_STOP_TIMEOUT + 0.1)

        self.assertEqual(len(stops), 1)
        self.assertTrue(stops[0].confirms_speculation)

    async def test_the_stop_timeout_waits_for_a_scheduled_decision(self):
        scheduled = ScheduledDecisionStopStrategy()
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(stop=[scheduled]),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )
        await controller.setup(frame_processor_setup(self.task_manager))
        await controller.start()
        stops = []

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            stops.append(strategy)

        await controller.process_frame(VADUserStartedSpeakingFrame())
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(USER_TURN_STOP_TIMEOUT * 2 + 0.1)
        self.assertEqual(stops, [])

        scheduled.scheduled = False
        await asyncio.sleep(USER_TURN_STOP_TIMEOUT + 0.1)
        self.assertEqual(stops, [None])

    async def _start_with_the_marker(self) -> tuple[list, list]:
        user = DecidingStopStrategy()
        marker = LLMTurnCompletionUserTurnStopStrategy(
            config=UserTurnCompletionConfig(incomplete_long_timeout=USER_TURN_STOP_TIMEOUT * 3)
        )
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(stop=[user, marker]),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )
        await controller.setup(frame_processor_setup(self.task_manager))
        await controller.start()
        stops, timeouts = [], []

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            stops.append(strategy)

        @controller.event_handler("on_user_turn_stop_timeout")
        async def on_user_turn_stop_timeout(controller):
            timeouts.append(None)

        await controller.process_frame(VADUserStartedSpeakingFrame())
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await user.decide()
        self.controller, self.marker = controller, marker
        return stops, timeouts

    async def test_a_turn_the_llm_needs_more_time_on_waits_for_the_marker(self):
        """After ○, the turn stays open until the marker strategy's own timeout decides."""
        stops, timeouts = await self._start_with_the_marker()
        inference = LLMContextFrame(context=LLMContext())
        await self.controller.process_frame(inference)
        await self.controller.process_frame(
            UserTurnCompletionFrame(kind="long", context_frame_id=inference.id)
        )

        await asyncio.sleep(USER_TURN_STOP_TIMEOUT * 2)
        self.assertEqual(stops, [])

        await asyncio.sleep(USER_TURN_STOP_TIMEOUT * 2)
        self.assertEqual(stops, [self.marker])
        self.assertEqual(timeouts, [])

    async def test_a_marker_that_never_comes_still_times_out(self):
        stops, timeouts = await self._start_with_the_marker()
        await self.controller.process_frame(LLMContextFrame(context=LLMContext()))

        await asyncio.sleep(USER_TURN_STOP_TIMEOUT + 0.1)
        self.assertEqual(stops, [None])
        self.assertEqual(timeouts, [None])

    async def test_external_turn_decided_while_vad_hears_speech_ends_once_vad_stops(self):
        """The service says the user is done while local VAD still hears a breath."""
        for stop_mode in (StrategyMode.ALL, StrategyMode.FIRST):
            with self.subTest(stop_mode=stop_mode):
                await self._decide_external_turn_while_vad_hears_speech(stop_mode)

    async def _decide_external_turn_while_vad_hears_speech(self, stop_mode: StrategyMode):
        controller = UserTurnController(
            user_turn_strategies=ExternalUserTurnStrategies(),
            user_turn_strategies_params=UserTurnStrategiesParams(stop_mode=stop_mode),
            user_turn_stop_timeout=3.0,
        )
        await controller.setup(frame_processor_setup(self.task_manager))
        await controller.start()
        stops, timeouts = [], []

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            stops.append(strategy)

        @controller.event_handler("on_user_turn_stop_timeout")
        async def on_user_turn_stop_timeout(controller):
            timeouts.append(None)

        await controller.process_frame(UserStartedSpeakingFrame())
        await controller.process_frame(UserStoppedSpeakingFrame())
        await controller.process_frame(VADUserStartedSpeakingFrame())
        await controller.process_frame(
            TranscriptionFrame(text="Hello", user_id="", timestamp="now")
        )
        # The transcription settles, and the strategy decides while VAD still
        # hears speech.
        await asyncio.sleep(0.7)
        self.assertEqual(stops, [])

        await controller.process_frame(VADUserStoppedSpeakingFrame())

        self.assertEqual(len(stops), 1)
        self.assertEqual(timeouts, [])
        await controller.cleanup()

    async def test_external_user_turn_strategies_no_timeout_while_speaking(self):
        """Test that timeout does not trigger when user is still speaking with external strategies."""
        controller = UserTurnController(
            user_turn_strategies=ExternalUserTurnStrategies(),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )

        await controller.setup(frame_processor_setup(self.task_manager))
        await controller.start()

        should_start = None
        should_stop = None
        timeout = None

        @controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            nonlocal should_start
            should_start = True

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            nonlocal should_stop
            should_stop = True

        @controller.event_handler("on_user_turn_stop_timeout")
        async def on_user_turn_stop_timeout(controller):
            nonlocal timeout
            timeout = True

        # Simulate external service (like Deepgram Flux) broadcasting UserStartedSpeakingFrame
        await controller.process_frame(UserStartedSpeakingFrame())
        self.assertTrue(should_start)
        self.assertFalse(should_stop)
        self.assertFalse(timeout)

        # User is still speaking, timeout should not trigger
        await asyncio.sleep(USER_TURN_STOP_TIMEOUT + 0.1)
        self.assertTrue(should_start)
        self.assertFalse(should_stop)
        self.assertFalse(timeout)

        # Now external service broadcasts UserStoppedSpeakingFrame
        await controller.process_frame(UserStoppedSpeakingFrame())

        # But no transcription, so timeout should trigger
        await asyncio.sleep(USER_TURN_STOP_TIMEOUT + 0.1)

        self.assertTrue(should_start)
        self.assertTrue(should_stop)
        self.assertTrue(timeout)

    async def test_proposed_frames_drive_the_turn_and_emit(self):
        """A proposal leaves the decision to the strategy, which emits."""
        controller = UserTurnController(
            user_turn_strategies=ExternalUserTurnStrategies(),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )
        await controller.setup(frame_processor_setup(self.task_manager))
        await controller.start()

        started_params = None
        stopped_params = None

        @controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            nonlocal started_params
            started_params = params

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            nonlocal stopped_params
            stopped_params = params

        await controller.process_frame(ProposedUserStartedSpeakingFrame())
        self.assertTrue(started_params.enable_user_speaking_frames)
        self.assertTrue(started_params.enable_interruptions)

        await controller.process_frame(
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now")
        )
        await controller.process_frame(ProposedUserStoppedSpeakingFrame())
        self.assertTrue(stopped_params.enable_user_speaking_frames)

        await controller.cleanup()

    async def test_real_turn_frames_drive_the_turn_but_suppress_emission(self):
        """The emitter already announced the turn, so the strategy emits nothing."""
        controller = UserTurnController(
            user_turn_strategies=ExternalUserTurnStrategies(),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )
        await controller.setup(frame_processor_setup(self.task_manager))
        await controller.start()

        started_params = None
        stopped_params = None

        @controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            nonlocal started_params
            started_params = params

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            nonlocal stopped_params
            stopped_params = params

        await controller.process_frame(UserStartedSpeakingFrame())
        self.assertFalse(started_params.enable_user_speaking_frames)
        self.assertFalse(started_params.enable_interruptions)

        await controller.process_frame(
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now")
        )
        await controller.process_frame(UserStoppedSpeakingFrame())
        self.assertFalse(stopped_params.enable_user_speaking_frames)

        await controller.cleanup()

    async def test_proposal_arriving_mid_speech_does_not_stop_the_turn(self):
        """The finalization guard still applies when proposals drive the turn."""
        controller = UserTurnController(
            user_turn_strategies=ExternalUserTurnStrategies(),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )
        await controller.setup(frame_processor_setup(self.task_manager))
        await controller.start()

        stopped = False

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            nonlocal stopped
            stopped = True

        # The controller tracks _user_speaking from the proposals, so a
        # completion resolving mid-speech is still dropped as stale.
        await controller.process_frame(ProposedUserStartedSpeakingFrame())
        await controller.process_frame(UserTurnInferenceCompletedFrame())
        self.assertFalse(stopped)

        await controller.cleanup()

    async def test_proposals_are_resolved_only_by_resolving_strategies(self):
        """A controller whose strategies can't resolve a proposal reports so."""
        external = UserTurnController(user_turn_strategies=ExternalUserTurnStrategies())
        self.assertTrue(external.resolves_proposed_turn_start_frames)
        self.assertTrue(external.resolves_proposed_turn_stop_frames)

        default = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)],
            )
        )
        self.assertFalse(default.resolves_proposed_turn_start_frames)
        self.assertFalse(default.resolves_proposed_turn_stop_frames)

    async def test_each_side_reports_its_own_proposal_resolution(self):
        """Resolving turn starts says nothing about turn stops, and vice versa."""
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[ExternalUserTurnStartStrategy()],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)],
            )
        )
        self.assertTrue(controller.resolves_proposed_turn_start_frames)
        self.assertFalse(controller.resolves_proposed_turn_stop_frames)

    async def test_custom_strategies_can_resolve_proposals(self):
        """Resolution follows what a strategy declares, not what class it is."""

        class CustomStartStrategy(BaseUserTurnStartStrategy):
            @property
            def resolves_proposed_turn_start_frames(self) -> bool:
                return True

        class CustomStopStrategy(BaseUserTurnStopStrategy):
            @property
            def resolves_proposed_turn_stop_frames(self) -> bool:
                return True

        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[CustomStartStrategy()],
                stop=[CustomStopStrategy()],
            )
        )
        self.assertTrue(controller.resolves_proposed_turn_start_frames)
        self.assertTrue(controller.resolves_proposed_turn_stop_frames)

    async def test_deferring_a_strategy_keeps_its_proposal_resolution(self):
        """Deferred finalization changes when the turn ends, not who decides it."""
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                stop=[deferred(ExternalUserTurnStopStrategy())],
            )
        )
        self.assertTrue(controller.resolves_proposed_turn_stop_frames)

    async def test_late_transcription_between_turns_no_premature_stop(self):
        """Test that a late transcription arriving between turns does not cause a premature stop.

        Reproduces the bug from issue #4053: after turn 1 completes and reset()
        clears state, a late TranscriptionFrame sets _text to stale content. On
        the next turn, that stale _text gates a premature turn stop via timeout(0)
        before the current turn's transcript arrives.

        Uses only VADUserTurnStartStrategy (no TranscriptionUserTurnStartStrategy)
        so the late transcription doesn't trigger a spurious turn start.
        """
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)],
            ),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )

        await controller.setup(frame_processor_setup(self.task_manager))
        await controller.start()

        start_count = 0
        stop_count = 0

        @controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            nonlocal start_count
            start_count += 1

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            nonlocal stop_count
            stop_count += 1

        # === Turn 1: S-T-E ===
        await controller.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(start_count, 1)

        await controller.process_frame(
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now")
        )

        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(TRANSCRIPTION_TIMEOUT + 0.1)
        self.assertEqual(stop_count, 1)

        # === Between turns: late transcription arrives ===
        # This sets _text on the stop strategy while _user_turn is False.
        await controller.process_frame(
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now")
        )

        # === Turn 2: S-T-E (transcription arrives during turn) ===
        # The fix resets stop strategies at turn start, clearing stale _text.
        await controller.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(start_count, 2)

        await controller.process_frame(
            TranscriptionFrame(text="How are you?", user_id="", timestamp="now")
        )

        await controller.process_frame(VADUserStoppedSpeakingFrame())

        # Wait for user_speech_timeout to elapse — should get turn 2 stop
        await asyncio.sleep(TRANSCRIPTION_TIMEOUT + 0.1)
        self.assertEqual(stop_count, 2)

    async def test_turn_start_notifies_start_and_stop_strategies(self):
        """Turn start calls handle_user_turn_started on both start and stop strategies."""
        started = []

        class SpyStart(VADUserTurnStartStrategy):
            async def handle_user_turn_started(self):
                started.append("start")

        class SpyStop(SpeechTimeoutUserTurnStopStrategy):
            async def handle_user_turn_started(self):
                await super().handle_user_turn_started()
                started.append("stop")

        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[SpyStart()],
                stop=[SpyStop(user_speech_timeout=TRANSCRIPTION_TIMEOUT)],
            ),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )
        await controller.setup(frame_processor_setup(self.task_manager))
        await controller.start()

        self.assertEqual(started, [])
        await controller.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(started, ["start", "stop"])

        await controller.cleanup()

    async def test_handle_user_turn_stopped_called_on_strategy_stop(self):
        """handle_user_turn_stopped() runs on a stop strategy when it ends the turn, never at start."""
        finalized = 0

        class SpyStopStrategy(SpeechTimeoutUserTurnStopStrategy):
            async def handle_user_turn_stopped(self):
                nonlocal finalized
                await super().handle_user_turn_stopped()
                finalized += 1

        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[SpyStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)],
            ),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )
        await controller.setup(frame_processor_setup(self.task_manager))
        await controller.start()

        await controller.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(finalized, 0)

        await controller.process_frame(
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now")
        )
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(TRANSCRIPTION_TIMEOUT + 0.1)
        self.assertEqual(finalized, 1)

        await controller.cleanup()

    async def test_handle_user_turn_stopped_called_on_watchdog_stop(self):
        """handle_user_turn_stopped() also runs when the stop watchdog ends the turn."""
        finalized = 0

        class SpyNeverStopStrategy(BaseUserTurnStopStrategy):
            async def handle_user_turn_stopped(self):
                nonlocal finalized
                await super().handle_user_turn_stopped()
                finalized += 1

        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[SpyNeverStopStrategy()],
            ),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )
        await controller.setup(frame_processor_setup(self.task_manager))
        await controller.start()

        await controller.process_frame(VADUserStartedSpeakingFrame())
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        self.assertEqual(finalized, 0)

        # No completion ever arrives; the watchdog finalizes the turn.
        await asyncio.sleep(USER_TURN_STOP_TIMEOUT + 0.1)
        self.assertEqual(finalized, 1)

        await controller.cleanup()

    async def test_start_strategy_not_reset_on_turn_stop(self):
        """A start strategy's handle_user_turn_stopped is a no-op (the deliberate asymmetry).

        Start strategies are armed on turn start but never reset on turn stop:
        their reset is turn-start semantic (e.g. WakePhrase refreshes its
        keepalive timeout), so a stop-side reset would be wrong.
        """
        resets = 0

        with self.assertWarns(DeprecationWarning):

            class LegacyStart(VADUserTurnStartStrategy):
                async def reset(self):  # the deprecated hook
                    nonlocal resets
                    resets += 1

        strategy = LegacyStart()

        # Arming on turn start bridges to reset()...
        await strategy.handle_user_turn_started()
        self.assertEqual(resets, 1)

        # ...but turn stop does not touch it.
        await strategy.handle_user_turn_stopped()
        self.assertEqual(resets, 1)

    async def test_stop_strategy_reset_bridged_on_both_callbacks(self):
        """A stop strategy's reset() is bridged on both turn boundaries (armed, then cleaned)."""
        resets = 0

        with self.assertWarns(DeprecationWarning):

            class LegacyStop(BaseUserTurnStopStrategy):
                async def reset(self):  # the deprecated hook
                    nonlocal resets
                    resets += 1

        strategy = LegacyStop()
        await strategy.handle_user_turn_started()
        await strategy.handle_user_turn_stopped()
        self.assertEqual(resets, 2)

    async def test_overriding_reset_warns_at_class_definition(self):
        """Overriding the deprecated reset() warns when the class is defined."""
        with self.assertWarns(DeprecationWarning):

            class LegacyStop(BaseUserTurnStopStrategy):
                async def reset(self):
                    pass

    async def test_reset_warning_shows_under_default_filters_at_the_subclass(self):
        """The reset() warning survives the default DeprecationWarning filter and names the subclass's file."""
        for base in (BaseUserTurnStartStrategy, BaseUserTurnStopStrategy):
            with self.subTest(base=base.__name__):
                with warnings.catch_warnings(record=True) as caught:
                    # Python's default filter for a module other than __main__.
                    warnings.simplefilter("ignore", DeprecationWarning)

                    class Legacy(base):
                        async def reset(self):
                            pass

                self.assertEqual(len(caught), 1)
                self.assertIs(caught[0].category, DeprecationWarning)
                self.assertEqual(caught[0].filename, __file__)

    async def test_subclass_of_concrete_strategy_overriding_reset_warns_but_is_not_bridged(self):
        """Subclassing a concrete strategy and overriding reset() warns, but reset() won't run.

        A concrete strategy's callback overrides don't route through the
        backward-compat bridge, so a legacy reset() on such a subclass is never
        invoked. __init_subclass__ still flags it loudly so the author migrates
        to the callbacks — no silent no-op.
        """
        reset_calls = 0

        with self.assertWarns(DeprecationWarning):

            class MyMinWords(MinWordsUserTurnStartStrategy):
                async def reset(self):
                    nonlocal reset_calls
                    reset_calls += 1

        strategy = MyMinWords(min_words=3)
        await strategy.handle_user_turn_started()
        self.assertEqual(reset_calls, 0)

    async def test_overriding_callbacks_does_not_warn_at_class_definition(self):
        """Defining a strategy that overrides the callbacks (not reset) doesn't warn."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            class Migrated(BaseUserTurnStopStrategy):
                async def handle_user_turn_stopped(self):
                    pass

    async def test_migrated_strategy_does_not_warn_at_runtime(self):
        """A strategy overriding the callbacks never warns on the per-turn callbacks."""
        strategy = SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)
        await strategy.setup(frame_processor_setup(self.task_manager))
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            await strategy.handle_user_turn_started()
            await strategy.handle_user_turn_stopped()
        await strategy.cleanup()

    async def test_turn_analyzer_cleared_on_stop_but_not_start(self):
        """The analyzer is cleared when the turn ends, but not when it starts.

        Clearing on start would drop the continuously-fed pre-speech buffer;
        that's the whole reason the clear can't live in the shared reset.
        """
        analyzer = MagicMock()
        strategy = TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer)

        await strategy.handle_user_turn_started()
        analyzer.clear.assert_not_called()

        await strategy.handle_user_turn_stopped()
        analyzer.clear.assert_called_once()

    async def test_deferred_forwards_both_callbacks(self):
        """The deferred() wrapper forwards both callbacks to its inner strategy."""
        analyzer = MagicMock()
        strategy = deferred(TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer))

        await strategy.handle_user_turn_started()
        analyzer.clear.assert_not_called()

        await strategy.handle_user_turn_stopped()
        analyzer.clear.assert_called_once()

    def _make_turn_analyzer_strategy(self, *, inference_secs: float, **kwargs):
        """Build a TurnAnalyzerUserTurnStopStrategy whose analyzer takes real time.

        Returns (strategy, released) where ``released`` is an asyncio.Event set
        when the strategy fires on_user_turn_stopped.
        """
        analyzer = MagicMock()

        async def slow_analyze():
            await asyncio.sleep(inference_secs)
            return (EndOfTurnState.COMPLETE, None)

        analyzer.analyze_end_of_turn = slow_analyze
        analyzer.cleanup = AsyncMock()

        strategy = TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer, **kwargs)

        released = asyncio.Event()

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            released.set()

        return strategy, released

    async def test_stt_timeout_anchored_to_speech_end(self):
        """The STT safety-net timeout doesn't grow with analyzer inference time.

        The p99 budget is anchored at the end of the user's speech
        (VAD stop timestamp minus stop_secs). Time the analyzer spends in
        analyze_end_of_turn() must come out of that budget, not extend it.
        """
        inference_secs = 0.3
        stop_secs = 0.2
        stt_p99 = 0.6  # deadline: 0.4s after the VAD stop frame is created

        strategy, released = self._make_turn_analyzer_strategy(inference_secs=inference_secs)
        await strategy.setup(frame_processor_setup(self.task_manager))
        await strategy.process_frame(STTMetadataFrame(service_name="stt", ttfs_p99_latency=stt_p99))
        await strategy.process_frame(VADUserStartedSpeakingFrame(start_secs=stop_secs))

        start = asyncio.get_running_loop().time()
        vad_stop = VADUserStoppedSpeakingFrame(stop_secs=stop_secs)
        vad_task = asyncio.create_task(strategy.process_frame(vad_stop))
        # A non-finalized transcript arrives well within the budget.
        await strategy.process_frame(
            TranscriptionFrame(user_id="", text="hello", timestamp="now", finalized=False)
        )
        await vad_task

        await asyncio.wait_for(released.wait(), timeout=2.0)
        elapsed = asyncio.get_running_loop().time() - start

        # Unanchored, release would come at inference + (p99 - stop_secs) = 0.7s.
        # Anchored, it comes at p99 - stop_secs = 0.4s after the VAD stop frame.
        self.assertLess(elapsed, 0.55)
        self.assertGreaterEqual(elapsed, inference_secs)

        await strategy.cleanup()

    async def test_finalized_transcript_during_inference_releases_immediately(self):
        """A transcript that finalizes mid-inference releases the turn right after.

        Transcripts are data frames and VAD stop is a system frame, so they are
        processed on different tasks and can interleave with the
        analyze_end_of_turn() await. Once the analyzer's verdict lands, the
        already-finalized transcript must release the turn immediately instead
        of waiting out the safety-net timer.
        """
        inference_secs = 0.3
        stt_p99 = 5.0  # timer far out, so an immediate release is unambiguous

        strategy, released = self._make_turn_analyzer_strategy(inference_secs=inference_secs)
        await strategy.setup(frame_processor_setup(self.task_manager))
        await strategy.process_frame(STTMetadataFrame(service_name="stt", ttfs_p99_latency=stt_p99))
        await strategy.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2))

        vad_task = asyncio.create_task(
            strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
        )
        await asyncio.sleep(inference_secs / 3)  # land mid-inference
        await strategy.process_frame(
            TranscriptionFrame(user_id="", text="hello", timestamp="now", finalized=True)
        )
        await vad_task

        await asyncio.wait_for(released.wait(), timeout=1.0)

        await strategy.cleanup()

    async def test_no_wait_for_transcript_releases_on_verdict(self):
        """With wait_for_transcript=False the verdict alone releases the turn.

        No transcript is ever sent; release must follow analyze_end_of_turn()
        directly rather than the STT safety-net timer.
        """
        stt_p99 = 5.0

        strategy, released = self._make_turn_analyzer_strategy(
            inference_secs=0.1, wait_for_transcript=False
        )
        await strategy.setup(frame_processor_setup(self.task_manager))
        await strategy.process_frame(STTMetadataFrame(service_name="stt", ttfs_p99_latency=stt_p99))
        await strategy.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2))
        await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))

        await asyncio.wait_for(released.wait(), timeout=1.0)

        await strategy.cleanup()


class ScheduledDecisionStopStrategy(BaseUserTurnStopStrategy):
    """A stop strategy with a decision of its own scheduled while ``scheduled``."""

    def __init__(self):
        super().__init__()
        self.scheduled = True

    @property
    def decision_scheduled(self) -> bool:
        return self.scheduled


class DecidingStopStrategy(BaseUserTurnStopStrategy):
    """A stop strategy that decides the user is done when a test says so."""

    def __init__(
        self,
        result: ProcessFrameResult = ProcessFrameResult.CONTINUE,
        *,
        on_inference: bool = False,
    ):
        super().__init__()
        self.frames: list[Frame] = []
        self._result = result
        self._on_inference = on_inference

    @property
    def waits_for_inference(self) -> bool:
        return self._on_inference

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        self.frames.append(frame)
        return self._result

    async def decide(self, **kwargs):
        await self.report_user_turn_stopped(**kwargs)

    async def speculate(self):
        await self.start_user_turn_speculation(UserTurnSpeculation("Hi"))

    async def confirm(self):
        await self.report_user_turn_stopped(confirms_speculation=True)


class TestStopModeAll(unittest.IsolatedAsyncioTestCase):
    """By default the turn ends once every stop strategy has decided."""

    async def asyncSetUp(self):
        self.task_manager = TaskManager()
        self.stops: list[tuple[BaseUserTurnStopStrategy | None, UserTurnStoppedParams]] = []

    async def asyncTearDown(self):
        await self.controller.cleanup()

    async def _start(
        self,
        stop: list[BaseUserTurnStopStrategy],
        *,
        start: list[BaseUserTurnStartStrategy] | None = None,
        stop_mode: StrategyMode = StrategyMode.ALL,
    ):
        self.controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=start or [VADUserTurnStartStrategy()], stop=stop
            ),
            user_turn_strategies_params=UserTurnStrategiesParams(stop_mode=stop_mode),
        )
        await self.controller.setup(frame_processor_setup(self.task_manager))

        @self.controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            self.stops.append((strategy, params))

    async def _speak(self):
        await self.controller.process_frame(VADUserStartedSpeakingFrame())
        await self.controller.process_frame(VADUserStoppedSpeakingFrame())

    async def test_the_turn_ends_once_every_strategy_decides(self):
        first, last = DecidingStopStrategy(), DecidingStopStrategy()
        await self._start([first, last])
        await self._speak()

        await first.decide()
        self.assertEqual(self.stops, [])

        await last.decide()
        self.assertEqual([s for s, _ in self.stops], [last])

    async def test_the_first_decision_ends_the_turn_in_first_mode(self):
        first, last = DecidingStopStrategy(), DecidingStopStrategy()
        await self._start([first, last], stop_mode=StrategyMode.FIRST)
        await self._speak()

        await first.decide()
        await last.decide()

        self.assertEqual([s for s, _ in self.stops], [first])

    async def test_a_first_list_waits_for_the_decision_on_the_inference(self):
        first, on_inference = DecidingStopStrategy(), DecidingStopStrategy(on_inference=True)
        await self._start([first, on_inference], stop_mode=StrategyMode.FIRST)
        inferences = self._record_inferences()
        await self._speak()

        await first.decide()
        self.assertEqual(len(inferences), 1)
        self.assertEqual(self.stops, [])

        await on_inference.decide()
        self.assertEqual([s for s, _ in self.stops], [on_inference])

    async def test_a_first_list_keeps_a_confirmation_until_the_turn_ends(self):
        first, on_inference = DecidingStopStrategy(), DecidingStopStrategy(on_inference=True)
        await self._start([first, on_inference], stop_mode=StrategyMode.FIRST)
        await self._speak()

        await first.speculate()
        await first.confirm()
        await on_inference.decide()

        _, params = self.stops[0]
        self.assertTrue(params.confirms_speculation)

    async def test_new_speech_clears_the_decisions(self):
        first, last = DecidingStopStrategy(), DecidingStopStrategy()
        await self._start([first, last])
        await self._speak()
        await first.decide()

        await self._speak()
        await last.decide()
        self.assertEqual(self.stops, [])

        await first.decide()
        self.assertEqual([s for s, _ in self.stops], [first])

    async def test_a_decision_made_while_the_user_speaks_counts(self):
        first, last = DecidingStopStrategy(), DecidingStopStrategy()
        await self._start([first, last])
        await self.controller.process_frame(VADUserStartedSpeakingFrame())
        await first.decide()
        await self.controller.process_frame(VADUserStoppedSpeakingFrame())

        await last.decide()

        self.assertEqual([s for s, _ in self.stops], [last])

    async def test_the_turn_does_not_end_while_the_user_speaks(self):
        first, last = DecidingStopStrategy(), DecidingStopStrategy()
        await self._start([first, last])
        await self.controller.process_frame(VADUserStartedSpeakingFrame())

        await first.decide()
        await last.decide()

        self.assertEqual(self.stops, [])

    async def test_a_turn_decided_while_the_user_speaks_ends_once_they_stop(self):
        first, last = DecidingStopStrategy(), DecidingStopStrategy()
        await self._start([first, last])
        await self.controller.process_frame(VADUserStartedSpeakingFrame())
        await first.decide()
        await last.decide()

        await self.controller.process_frame(VADUserStoppedSpeakingFrame())

        self.assertEqual([s for s, _ in self.stops], [last])

    async def test_a_first_list_ends_a_turn_decided_while_the_user_speaks_once_they_stop(self):
        strategy = DecidingStopStrategy()
        await self._start([strategy], stop_mode=StrategyMode.FIRST)
        await self.controller.process_frame(VADUserStartedSpeakingFrame())
        await strategy.decide()
        self.assertEqual(self.stops, [])

        await self.controller.process_frame(VADUserStoppedSpeakingFrame())

        self.assertEqual([s for s, _ in self.stops], [strategy])

    async def test_a_cancelled_answer_voids_the_decisions_made_on_it(self):
        """A decision on the inference is about the answer, so it goes with it."""
        user, on_inference = DecidingStopStrategy(), DecidingStopStrategy(on_inference=True)
        await self._start([user, on_inference])
        await self._speak()
        await user.speculate()
        await on_inference.decide()

        await self.controller.process_frame(UserTurnResponseCancelFrame())
        await user.decide()

        self.assertEqual(self.stops, [])

    async def test_a_cancelled_answer_keeps_the_decisions_on_the_user(self):
        """A decision on the user says nothing about the answer, so it outlives it."""
        user, eager, on_inference = (
            DecidingStopStrategy(),
            DecidingStopStrategy(),
            DecidingStopStrategy(on_inference=True),
        )
        await self._start([user, eager, on_inference])
        inferences = self._record_inferences()
        await self._speak()
        await eager.speculate()
        await user.decide()

        await self.controller.process_frame(UserTurnResponseCancelFrame())
        await eager.decide()

        self.assertEqual([speculation is None for _, speculation in inferences], [False, True])

    async def test_a_confirmation_outlives_the_answer_it_confirmed(self):
        """The turn's answer runs again for a decision that confirmed a cancelled one."""
        confirming, on_inference = DecidingStopStrategy(), DecidingStopStrategy(on_inference=True)
        await self._start([confirming, on_inference])
        inferences = []

        @self.controller.event_handler("on_user_turn_inference_triggered")
        async def on_user_turn_inference_triggered(controller, strategy, speculation):
            inferences.append(strategy)

        await self._speak()
        await confirming.speculate()
        await confirming.confirm()
        await self.controller.process_frame(UserTurnResponseCancelFrame())
        self.assertEqual(inferences, [confirming, confirming])

        await on_inference.decide()
        _, params = self.stops[0]
        self.assertFalse(params.confirms_speculation)

    async def test_a_withdrawn_speculation_reaches_every_stop_strategy(self):
        withdrawing, other = DecidingStopStrategy(), DecidingStopStrategy()
        await self._start([withdrawing, other])
        await self._speak()

        await withdrawing.cancel_user_turn_speculation()

        self.assertIsInstance(other.frames[-1], UserTurnResponseCancelFrame)

    async def test_each_turn_needs_its_own_decisions(self):
        """Decisions don't carry over into a turn that starts without speech frames."""
        first, last = DecidingStopStrategy(), DecidingStopStrategy()
        await self._start([first, last], start=[TranscriptionUserTurnStartStrategy()])
        await self.controller.process_frame(TranscriptionFrame(text="Hi", user_id="", timestamp=""))
        await first.decide()
        await last.decide()

        await self.controller.process_frame(TranscriptionFrame(text="Hi", user_id="", timestamp=""))
        await last.decide()

        self.assertEqual(len(self.stops), 1)

    async def test_a_deferred_strategy_decides_as_the_one_it_wraps(self):
        inner, other = DecidingStopStrategy(), DecidingStopStrategy()
        await self._start([deferred(inner), other])
        await self._speak()

        await other.decide()
        self.assertEqual(self.stops, [])

        await inner.decide()
        self.assertEqual([s for s, _ in self.stops], [inner])

    async def test_every_strategy_sees_every_frame(self):
        blocking, other = DecidingStopStrategy(ProcessFrameResult.STOP), DecidingStopStrategy()
        await self._start([blocking, other])

        await self._speak()

        self.assertEqual(len(other.frames), 2)

    def _record_inferences(self) -> list[tuple[BaseUserTurnStopStrategy | None, object]]:
        inferences: list[tuple[BaseUserTurnStopStrategy | None, object]] = []

        @self.controller.event_handler("on_user_turn_inference_triggered")
        async def on_user_turn_inference_triggered(controller, strategy, speculation):
            inferences.append((strategy, speculation))

        return inferences

    async def test_the_inference_waits_for_every_strategy_that_listens_to_the_user(self):
        first, second = DecidingStopStrategy(), DecidingStopStrategy()
        on_inference = DecidingStopStrategy(on_inference=True)
        await self._start([first, second, on_inference])
        inferences = self._record_inferences()
        await self._speak()

        await first.decide()
        self.assertEqual(inferences, [])

        await second.decide()
        self.assertEqual([s for s, _ in inferences], [second])
        self.assertEqual(self.stops, [])

        await on_inference.decide()
        self.assertEqual([s for s, _ in self.stops], [on_inference])

    async def test_a_list_that_only_waits_for_the_inference_warns(self):
        """With no stop strategy listening to the user, only the stop timeout runs the inference."""
        messages = []
        sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
        try:
            await self._start([DecidingStopStrategy(on_inference=True)])
            await self.controller.update_strategies(
                UserTurnStrategies(
                    stop=[DecidingStopStrategy(), DecidingStopStrategy(on_inference=True)]
                )
            )
        finally:
            logger.remove(sink)

        warned = [m for m in messages if "waits for the turn's inference" in m]
        self.assertEqual(len(warned), 1)

    async def test_the_inference_runs_once_per_round_of_decisions(self):
        first, on_inference = DecidingStopStrategy(), DecidingStopStrategy(on_inference=True)
        await self._start([first, on_inference])
        inferences = self._record_inferences()
        await self._speak()

        await first.decide()
        await first.decide()
        self.assertEqual(len(inferences), 1)

        await self._speak()
        await first.decide()
        self.assertEqual(len(inferences), 2)

    async def test_a_speculation_is_the_turns_answer(self):
        eager, on_inference = DecidingStopStrategy(), DecidingStopStrategy(on_inference=True)
        await self._start([eager, on_inference])
        inferences = self._record_inferences()
        await self._speak()

        await eager.speculate()
        await eager.confirm()

        self.assertEqual([speculation is None for _, speculation in inferences], [False])

    async def test_a_withdrawn_speculation_lets_the_inference_run(self):
        eager, on_inference = DecidingStopStrategy(), DecidingStopStrategy(on_inference=True)
        await self._start([eager, on_inference])
        inferences = self._record_inferences()
        await self._speak()

        await eager.speculate()
        await eager.cancel_user_turn_speculation()
        await eager.decide()

        self.assertEqual([speculation is None for _, speculation in inferences], [False, True])

    def _record_resumes(self) -> list[UserTurnResumedParams]:
        resumes: list[UserTurnResumedParams] = []

        @self.controller.event_handler("on_user_turn_resumed")
        async def on_user_turn_resumed(controller, params):
            resumes.append(params)

        return resumes

    async def test_speaking_again_in_an_open_turn_resumes_it(self):
        await self._start([DecidingStopStrategy(), DecidingStopStrategy()])
        resumes = self._record_resumes()
        await self._speak()

        await self._speak()

        self.assertEqual(
            resumes,
            [UserTurnResumedParams(enable_interruptions=True, inference_triggered=False)],
        )

    async def test_a_resume_after_the_inference_was_triggered_says_so(self):
        first, on_inference = DecidingStopStrategy(), DecidingStopStrategy(on_inference=True)
        await self._start([first, on_inference])
        resumes = self._record_resumes()
        await self._speak()
        await first.decide()

        await self._speak()

        self.assertEqual([p.inference_triggered for p in resumes], [True])

    async def test_one_speech_reported_twice_resumes_the_turn_once(self):
        await self._start([DecidingStopStrategy(), DecidingStopStrategy()])
        resumes = self._record_resumes()
        await self._speak()

        await self.controller.process_frame(VADUserStartedSpeakingFrame())
        await self.controller.process_frame(UserStartedSpeakingFrame())

        self.assertEqual(len(resumes), 1)

    async def test_the_deprecated_finalized_trigger_still_decides(self):
        strategy = DecidingStopStrategy()
        await self._start([strategy])
        await self._speak()

        with self.assertWarns(DeprecationWarning):
            await strategy.trigger_user_turn_finalized()

        self.assertEqual([s for s, _ in self.stops], [strategy])

    async def test_the_turn_end_combines_the_decisions(self):
        """A turn end announced elsewhere, or confirming a speculation, applies to the whole turn."""
        announced, confirming = DecidingStopStrategy(), DecidingStopStrategy()
        await self._start([announced, confirming])
        await self._speak()

        await confirming.speculate()
        await announced.report_user_turn_stopped(enable_user_speaking_frames=False)
        await confirming.confirm()

        _, params = self.stops[0]
        self.assertFalse(params.enable_user_speaking_frames)
        self.assertTrue(params.confirms_speculation)

    async def test_confirming_a_withdrawn_speculation_runs_the_turns_answer(self):
        """A confirmation that crosses its speculation's withdrawal confirms nothing."""
        confirming, on_inference = DecidingStopStrategy(), DecidingStopStrategy(on_inference=True)
        await self._start([confirming, on_inference])
        inferences = []

        @self.controller.event_handler("on_user_turn_inference_triggered")
        async def on_user_turn_inference_triggered(controller, strategy, speculation):
            inferences.append(speculation)

        await self._speak()
        await confirming.speculate()
        await self.controller.process_frame(UserTurnResponseCancelFrame())
        await confirming.confirm()
        self.assertEqual([s is None for s in inferences], [False, True])

        await on_inference.decide()
        _, params = self.stops[0]
        self.assertFalse(params.confirms_speculation)


class DecidingStartStrategy(BaseUserTurnStartStrategy):
    """A start strategy that decides the user started when a test says so."""

    def __init__(self, result: ProcessFrameResult = ProcessFrameResult.CONTINUE, **kwargs):
        super().__init__(**kwargs)
        self.frames: list[Frame] = []
        self._result = result

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        self.frames.append(frame)
        return self._result

    async def decide(self, **kwargs):
        await self.report_user_turn_started(**kwargs)


class TestStartModeAll(unittest.IsolatedAsyncioTestCase):
    """With StrategyMode.ALL the turn starts once every start strategy has decided."""

    async def asyncSetUp(self):
        self.task_manager = TaskManager()
        self.starts: list[tuple[BaseUserTurnStartStrategy | None, UserTurnStartedParams]] = []
        self.stop = DecidingStopStrategy()

    async def asyncTearDown(self):
        await self.controller.cleanup()

    async def _start(
        self,
        start: list[BaseUserTurnStartStrategy],
        *,
        start_mode: StrategyMode = StrategyMode.ALL,
    ):
        self.controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(start=start, stop=[self.stop]),
            user_turn_strategies_params=UserTurnStrategiesParams(start_mode=start_mode),
        )
        await self.controller.setup(frame_processor_setup(self.task_manager))

        @self.controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            self.starts.append((strategy, params))

    async def test_the_turn_starts_once_every_strategy_decides(self):
        first, last = DecidingStartStrategy(), DecidingStartStrategy()
        await self._start([first, last])

        await first.decide()
        self.assertEqual(self.starts, [])

        await last.decide()
        self.assertEqual([s for s, _ in self.starts], [last])

    async def test_the_first_decision_starts_the_turn_in_first_mode(self):
        first, last = DecidingStartStrategy(), DecidingStartStrategy()
        await self._start([first, last], start_mode=StrategyMode.FIRST)

        await first.decide()

        self.assertEqual([s for s, _ in self.starts], [first])

    async def test_every_strategy_sees_every_frame(self):
        blocking, other = DecidingStartStrategy(ProcessFrameResult.STOP), DecidingStartStrategy()
        await self._start([blocking, other])

        await self.controller.process_frame(VADUserStartedSpeakingFrame())

        self.assertEqual(len(other.frames), 1)

    async def test_the_turn_start_combines_the_decisions(self):
        quiet, other = DecidingStartStrategy(), DecidingStartStrategy()
        await self._start([quiet, other])

        await quiet.decide(enable_interruptions=False)
        await other.decide()

        _, params = self.starts[0]
        self.assertFalse(params.enable_interruptions)

    async def test_a_first_list_still_honors_a_stop_with_a_warning(self):
        blocking, other = DecidingStartStrategy(ProcessFrameResult.STOP), DecidingStartStrategy()
        await self._start([blocking, other], start_mode=StrategyMode.FIRST)
        deprecation._warned_sites.clear()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await self.controller.process_frame(VADUserStartedSpeakingFrame())

        self.assertEqual(other.frames, [])
        self.assertEqual(
            [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)],
            [
                "`ProcessFrameResult` is deprecated since 1.15.0 and will be removed in "
                "2.0.0. Use `StrategyMode` instead."
            ],
        )

    async def test_a_wake_phrase_alone_in_a_first_list_does_not_warn(self):
        await self._start(
            [WakePhraseUserTurnStartStrategy(phrases=["hey pipecat"])],
            start_mode=StrategyMode.FIRST,
        )
        deprecation._warned_sites.clear()

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            await self.controller.process_frame(VADUserStartedSpeakingFrame())

    async def test_a_wake_phrase_in_an_all_list_does_not_warn(self):
        await self._start(
            [VADUserTurnStartStrategy(), WakePhraseUserTurnStartStrategy(phrases=["hey pipecat"])]
        )
        deprecation._warned_sites.clear()

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            await self.controller.process_frame(VADUserStartedSpeakingFrame())

    async def test_each_turn_needs_its_own_decisions(self):
        first, last = DecidingStartStrategy(), DecidingStartStrategy()
        await self._start([first, last])
        await first.decide()
        await last.decide()
        await self.stop.decide()

        await last.decide()

        self.assertEqual(len(self.starts), 1)


if __name__ == "__main__":
    unittest.main()
