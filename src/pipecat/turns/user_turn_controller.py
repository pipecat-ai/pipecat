#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module defines a controller for managing user turn lifecycle."""

import asyncio
from dataclasses import replace
from enum import Enum

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    UserTurnResponseCancelFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.turns.types import (
    ProcessFrameResult,
    UserTurnResumedParams,
    UserTurnSpeculation,
)
from pipecat.turns.user_start import (
    BaseUserTurnStartStrategy,
    UserTurnStartedParams,
)
from pipecat.turns.user_stop import (
    BaseUserTurnStopStrategy,
    DeferredUserTurnStopStrategy,
    UserTurnStoppedParams,
)
from pipecat.turns.user_turn_strategies import (
    StrategyMode,
    UserTurnStrategies,
    UserTurnStrategiesParams,
)
from pipecat.utils.base_object import BaseObject
from pipecat.utils.deprecation import warn_deprecated


class _UserTurnInference(Enum):
    """The inference triggered for the stop decisions made so far in a user turn."""

    NONE = "none"
    TRIGGERED = "triggered"
    SPECULATIVE = "speculative"


class UserTurnController(BaseObject):
    """Controller for managing user turn lifecycle.

    This class manages user turn state (active/inactive), handles start and stop
    strategies, and emits events when user turns begin, end, or timeout occurs.

    How a turn starts and ends follows the ``start_mode`` and ``stop_mode`` of
    its :class:`~pipecat.turns.user_turn_strategies.UserTurnStrategiesParams`.
    With ``StrategyMode.FIRST`` the first strategy to decide starts or ends
    the turn. With ``StrategyMode.ALL`` the turn starts or ends once every
    strategy in the list has decided. Turns start on the first start
    strategy and end once every stop strategy agrees by default. A stop
    strategy that waits for the turn's inference gets its say in either
    mode: in a ``FIRST`` list, the first stop strategy to decide runs the
    inference, and the turn ends once a strategy waiting for that inference
    decides. A cancelled answer voids the stop decisions made on it.

    In an ``ALL`` list the controller starts the turn's inference itself,
    once every stop strategy that listens to the user has decided. A stop
    strategy that waits for the inference instead, such as one reading the
    LLM's turn-completion marker, then decides on that inference. An eager
    end of turn still starts its speculation at once.

    Event handlers available:

    - on_user_turn_started: Emitted when a user turn starts.
    - on_user_turn_inference_triggered: Emitted when enough signal exists to
      start LLM inference, carrying a `UserTurnSpeculation` when the turn it
      answers hasn't ended yet. Fires together with `on_user_turn_stopped` for
      most strategies; fires alone when the turn's end waits for a decision
      on the inference, or for an eager end of turn to be confirmed.
    - on_user_turn_stopped: Emitted when a user turn is semantically final.
    - on_user_turn_resumed: Emitted when the user starts speaking again in
      an open turn, after they stopped or after the turn's inference was
      triggered. Its `UserTurnResumedParams` say whether that inference
      was triggered, since the new speech voids the answer it started.
    - on_user_turn_stop_timeout: Emitted when the stop timeout ends a user turn
      its stop strategies didn't end.
    - on_user_turn_speculation_cancelled: Emitted when a stop strategy
      withdraws its speculative inference, such as an eager end of turn whose
      prediction turned out wrong.
    - on_reset_aggregation: Emitted when a start strategy wants the user's
      aggregated text dropped, as the wake phrase strategy does for speech
      before the wake phrase.
    - on_push_frame: Emitted when a strategy wants to push a frame.
    - on_broadcast_frame: Emitted when a strategy wants to broadcast a frame.

    Example::

        @controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy: BaseUserTurnStartStrategy, params: UserTurnStartedParams):
            ...

        @controller.event_handler("on_user_turn_inference_triggered")
        async def on_user_turn_inference_triggered(
            controller,
            strategy: BaseUserTurnStopStrategy,
            speculation: UserTurnSpeculation | None,
        ):
            ...

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy: BaseUserTurnStopStrategy, params: UserTurnStoppedParams):
            ...

        @controller.event_handler("on_user_turn_resumed")
        async def on_user_turn_resumed(controller, params: UserTurnResumedParams):
            ...

        @controller.event_handler("on_user_turn_stop_timeout")
        async def on_user_turn_stop_timeout(controller):
            ...

        @controller.event_handler("on_user_turn_speculation_cancelled")
        async def on_user_turn_speculation_cancelled(controller):
            ...

        @controller.event_handler("on_reset_aggregation")
        async def on_reset_aggregation(controller, strategy: BaseUserTurnStartStrategy):
            ...

        @controller.event_handler("on_push_frame")
        async def on_push_frame(controller, frame: Frame, direction: FrameDirection):
            ...

        @controller.event_handler("on_broadcast_frame")
        async def on_broadcast_frame(controller, frame_cls: Type[Frame], **kwargs):
            ...
    """

    def __init__(
        self,
        *,
        user_turn_strategies: UserTurnStrategies,
        user_turn_strategies_params: UserTurnStrategiesParams | None = None,
        user_turn_stop_timeout: float = 5.0,
    ):
        """Initialize the user turn controller.

        Args:
            user_turn_strategies: Configured strategies for starting and stopping user turns.
            user_turn_strategies_params: How the strategies decide. Defaults to
                :class:`~pipecat.turns.user_turn_strategies.UserTurnStrategiesParams`.
            user_turn_stop_timeout: Seconds without user activity before an open
                user turn is ended, unless a stop strategy has a decision scheduled.
        """
        super().__init__()

        self._user_turn_strategies = user_turn_strategies
        self._user_turn_stop_strategies = self._stop_strategies_of(user_turn_strategies)
        self._user_turn_strategies_params = (
            user_turn_strategies_params or UserTurnStrategiesParams()
        )
        self._user_turn_stop_timeout = user_turn_stop_timeout
        self._setup: FrameProcessorSetup | None = None

        self._user_speaking = False

        self._user_turn = False
        self._user_turn_start_decisions: dict[BaseUserTurnStartStrategy, UserTurnStartedParams] = {}
        self._user_turn_stop_decisions: dict[BaseUserTurnStopStrategy, UserTurnStoppedParams] = {}
        # The inference triggered for the stop decisions made so far, if any.
        self._user_turn_inference = _UserTurnInference.NONE
        # Whether the open turn interrupts the bot, as its start decided.
        self._user_turn_enable_interruptions = False
        self._user_turn_stop_timeout_event = asyncio.Event()
        self._user_turn_stop_timeout_task: asyncio.Task | None = None

        self._register_event_handler("on_push_frame", sync=True)
        self._register_event_handler("on_broadcast_frame", sync=True)
        self._register_event_handler("on_user_turn_started", sync=True)
        self._register_event_handler("on_user_turn_inference_triggered", sync=True)
        self._register_event_handler("on_user_turn_speculation_cancelled", sync=True)
        self._register_event_handler("on_user_turn_stopped", sync=True)
        self._register_event_handler("on_user_turn_resumed", sync=True)
        self._register_event_handler("on_user_turn_stop_timeout", sync=True)
        self._register_event_handler("on_reset_aggregation", sync=True)

    @property
    def user_turn_strategies(self) -> UserTurnStrategies:
        """The currently active user turn strategies."""
        return self._user_turn_strategies

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the controller.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup.task_manager)

        # Kept so update_strategies() can set up new strategies without the
        # caller having to hand us the setup again.
        self._setup = setup

        await self._setup_strategies()

    async def start(self):
        """Start watching for a turn that stops without the user stopping.

        Paired with :meth:`stop`.
        """
        if not self._user_turn_stop_timeout_task:
            self._user_turn_stop_timeout_task = self.create_task(
                self._user_turn_stop_timeout_task_handler()
            )

    async def stop(self):
        """Stop the turn stop timeout, leaving the strategies alone.

        Called at session end. The strategies may be shared, so cleaning
        them up waits for :meth:`cleanup`.
        """
        if self._user_turn_stop_timeout_task:
            await self.cancel_task(self._user_turn_stop_timeout_task)
            self._user_turn_stop_timeout_task = None

    async def cleanup(self):
        """Cleanup the controller."""
        await super().cleanup()

        await self.stop()

        await self._cleanup_strategies()

    async def update_strategies(self, strategies: UserTurnStrategies):
        """Replace the current strategies with the given ones.

        Args:
            strategies: The new user turn strategies the controller should use.
        """
        await self._cleanup_strategies()
        self._user_turn_strategies = strategies
        self._user_turn_stop_strategies = self._stop_strategies_of(strategies)
        self._user_turn_start_decisions.clear()
        self._user_turn_stop_decisions.clear()
        await self._setup_strategies()

    @property
    def resolves_proposed_turn_start_frames(self) -> bool:
        """Whether any active start strategy resolves proposed turn starts.

        A proposal is resolved once, so a caller holding this controller should stop
        forwarding :class:`~pipecat.frames.frames.ProposedUserStartedSpeakingFrame`
        when this is True — passing it along would let a resolver further down
        the pipeline decide the same turn a second time.
        """
        return any(
            s.resolves_proposed_turn_start_frames for s in self._user_turn_strategies.start or []
        )

    @property
    def resolves_proposed_turn_stop_frames(self) -> bool:
        """Whether any active stop strategy resolves proposed turn stops.

        The end-of-turn counterpart to :attr:`resolves_proposed_turn_start_frames`.
        """
        return any(s.resolves_proposed_turn_stop_frames for s in self._user_turn_stop_strategies)

    async def process_frame(self, frame: Frame):
        """Process an incoming frame to detect user turn start or stop.

        The frame is passed to the configured user turn strategies, which are
        responsible for deciding when a user turn starts or stops and emitting
        the corresponding events.

        Args:
            frame: The frame to be processed.

        """
        if isinstance(frame, (UserStartedSpeakingFrame, ProposedUserStartedSpeakingFrame)):
            await self._handle_user_started_speaking(frame)
        elif isinstance(frame, (UserStoppedSpeakingFrame, ProposedUserStoppedSpeakingFrame)):
            await self._handle_user_stopped_speaking(frame)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._handle_vad_user_started_speaking(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._handle_vad_user_stopped_speaking(frame)
        elif isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            await self._handle_transcription(frame)

        start_strategies = self._user_turn_strategies.start or []
        for strategy in start_strategies:
            result = await strategy.process_frame(frame)
            # Every strategy in an ALL list needs every frame to decide on.
            # The last strategy's STOP hides the frame from no one.
            if result == ProcessFrameResult.STOP and self._start_mode == StrategyMode.FIRST:
                if strategy is not start_strategies[-1]:
                    self._warn_process_frame_result()
                break

        stop_strategies = self._user_turn_stop_strategies
        for strategy in stop_strategies:
            result = await strategy.process_frame(frame)
            # Every strategy in an ALL list needs every frame to decide on.
            # The last strategy's STOP hides the frame from no one.
            if result == ProcessFrameResult.STOP and self._stop_mode == StrategyMode.FIRST:
                if strategy is not stop_strategies[-1]:
                    self._warn_process_frame_result()
                break

        # Handled once the stop strategies have seen it, so an inference started
        # here is the one they then wait for.
        if isinstance(frame, UserTurnResponseCancelFrame):
            await self._handle_user_turn_response_cancel()

    def _warn_process_frame_result(self):
        warn_deprecated(
            "`ProcessFrameResult` is deprecated since 1.15.0 and will be removed in 2.0.0. "
            "Use `StrategyMode` instead."
        )

    async def _setup_strategies(self):
        if not self._setup:
            raise RuntimeError(f"{self} was not properly set up")

        for s in self._user_turn_strategies.start or []:
            await s.setup(self._setup)
            s.add_event_handler("on_push_frame", self._on_push_frame)
            s.add_event_handler("on_broadcast_frame", self._on_broadcast_frame)
            s.add_event_handler("on_user_turn_started", self._on_user_turn_started)
            s.add_event_handler("on_reset_aggregation", self._on_reset_aggregation)

        for s in self._user_turn_stop_strategies:
            await s.setup(self._setup)
            s.add_event_handler("on_push_frame", self._on_push_frame)
            s.add_event_handler("on_broadcast_frame", self._on_broadcast_frame)
            s.add_event_handler(
                "on_user_turn_inference_triggered", self._on_user_turn_inference_triggered
            )
            s.add_event_handler(
                "on_user_turn_speculation_cancelled", self._on_user_turn_speculation_cancelled
            )
            s.add_event_handler("on_user_turn_stopped", self._on_user_turn_stopped)

        self._warn_if_no_stop_strategy_listens_to_the_user()

    def _warn_if_no_stop_strategy_listens_to_the_user(self):
        # The turn's inference runs once the stop strategies that listen to the
        # user have decided, so with none of them only the stop timeout runs it.
        stop_strategies = self._user_turn_stop_strategies
        if stop_strategies and all(s.waits_for_inference for s in stop_strategies):
            logger.warning(
                f"{self}: every user turn stop strategy waits for the turn's inference, so "
                "nothing runs it until the user turn stop timeout. Add a stop strategy that "
                "detects the user stopping, such as SpeechTimeoutUserTurnStopStrategy."
            )

    async def _cleanup_strategies(self):
        # Remove the handlers _setup_strategies added (symmetric), so re-applying
        # strategies via update_strategies — possibly reusing the same strategy
        # instances — doesn't accumulate duplicate handler registrations.
        for s in self._user_turn_strategies.start or []:
            await s.cleanup()
            s.remove_event_handler("on_push_frame", self._on_push_frame)
            s.remove_event_handler("on_broadcast_frame", self._on_broadcast_frame)
            s.remove_event_handler("on_user_turn_started", self._on_user_turn_started)
            s.remove_event_handler("on_reset_aggregation", self._on_reset_aggregation)

        for s in self._user_turn_stop_strategies:
            await s.cleanup()
            s.remove_event_handler("on_push_frame", self._on_push_frame)
            s.remove_event_handler("on_broadcast_frame", self._on_broadcast_frame)
            s.remove_event_handler(
                "on_user_turn_inference_triggered", self._on_user_turn_inference_triggered
            )
            s.remove_event_handler(
                "on_user_turn_speculation_cancelled", self._on_user_turn_speculation_cancelled
            )
            s.remove_event_handler("on_user_turn_stopped", self._on_user_turn_stopped)

    async def _on_push_frame(
        self,
        strategy: BaseUserTurnStartStrategy | BaseUserTurnStopStrategy,
        frame: Frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ):
        await self._call_event_handler("on_push_frame", frame, direction)

    async def _on_broadcast_frame(
        self,
        strategy: BaseUserTurnStartStrategy | BaseUserTurnStopStrategy,
        frame_cls: type[Frame],
        **kwargs,
    ):
        await self._call_event_handler("on_broadcast_frame", frame_cls, **kwargs)

    async def _handle_user_started_speaking(
        self, frame: UserStartedSpeakingFrame | ProposedUserStartedSpeakingFrame
    ):
        await self._handle_speech_started()

        # The user started talking, let's reset the user turn timeout.
        self._user_turn_stop_timeout_event.set()

    async def _handle_user_stopped_speaking(
        self, frame: UserStoppedSpeakingFrame | ProposedUserStoppedSpeakingFrame
    ):
        self._user_speaking = False

        # The user stopped talking, let's reset the user turn timeout.
        self._user_turn_stop_timeout_event.set()

        await self._end_user_turn_if_decided()

    async def _handle_vad_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        await self._handle_speech_started()

        # The user started talking, let's reset the user turn timeout.
        self._user_turn_stop_timeout_event.set()

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        self._user_speaking = False

        # The user stopped talking, let's reset the user turn timeout.
        self._user_turn_stop_timeout_event.set()

        await self._end_user_turn_if_decided()

    async def _handle_speech_started(self):
        # Speech in an open turn clears the stop decisions made so far, and
        # the inference triggered for them. It resumes the turn when the user had
        # stopped, or when that inference was already triggered.
        resumed = self._user_turn and (
            not self._user_speaking or self._user_turn_inference_triggered
        )
        params = UserTurnResumedParams(
            enable_interruptions=self._user_turn_enable_interruptions,
            inference_triggered=self._user_turn_inference_triggered,
        )
        self._user_speaking = True
        self._user_turn_stop_decisions.clear()
        self._user_turn_inference = _UserTurnInference.NONE
        if resumed:
            await self._call_event_handler("on_user_turn_resumed", params)

    async def _handle_transcription(self, frame: TranscriptionFrame | InterimTranscriptionFrame):
        # We have received a transcription, let's reset the user turn timeout.
        self._user_turn_stop_timeout_event.set()

    @property
    def _start_mode(self) -> StrategyMode:
        return self._user_turn_strategies_params.start_mode

    async def _on_user_turn_started(
        self,
        strategy: BaseUserTurnStartStrategy,
        params: UserTurnStartedParams,
    ):
        if self._start_mode == StrategyMode.ALL:
            await self._add_user_turn_start_decision(strategy, params)
        else:
            await self._trigger_user_turn_start(strategy, params)

    async def _on_reset_aggregation(self, strategy: BaseUserTurnStartStrategy):
        await self._call_event_handler("on_reset_aggregation", strategy)

    async def _add_user_turn_start_decision(
        self, strategy: BaseUserTurnStartStrategy, params: UserTurnStartedParams
    ):
        if self._user_turn:
            return

        self._user_turn_start_decisions[strategy] = params

        if len(self._user_turn_start_decisions) == len(self._user_turn_strategies.start or []):
            await self._trigger_user_turn_start(strategy, self._user_turn_start_params())

    def _user_turn_start_params(self) -> UserTurnStartedParams:
        # One decision saying the turn shouldn't interrupt, or was announced
        # elsewhere, applies to the whole turn.
        decisions = self._user_turn_start_decisions.values()
        return UserTurnStartedParams(
            enable_interruptions=all(d.enable_interruptions for d in decisions),
            enable_user_speaking_frames=all(d.enable_user_speaking_frames for d in decisions),
        )

    async def _trigger_user_turn_start(
        self, strategy: BaseUserTurnStartStrategy | None, params: UserTurnStartedParams
    ):
        # Prevent two consecutive user turn starts.
        if self._user_turn:
            return

        self._user_turn = True
        self._user_turn_start_decisions.clear()
        self._user_turn_stop_timeout_event.set()
        self._user_turn_enable_interruptions = params.enable_interruptions

        # Notify every strategy that the turn has started. Start strategies
        # ready themselves for the next detection; stop strategies arm to detect
        # this turn's end. A strategy resets whatever per-turn state it keeps
        # inside its own handle_user_turn_started.
        for s in self._user_turn_strategies.start or []:
            await s.handle_user_turn_started()
        for s in self._user_turn_stop_strategies:
            await s.handle_user_turn_started()

        await self._call_event_handler("on_user_turn_started", strategy, params)

    @property
    def _stop_mode(self) -> StrategyMode:
        return self._user_turn_strategies_params.stop_mode

    @property
    def _user_turn_inference_triggered(self) -> bool:
        return self._user_turn_inference != _UserTurnInference.NONE

    @staticmethod
    def _stop_strategies_of(strategies: UserTurnStrategies) -> list[BaseUserTurnStopStrategy]:
        # The controller runs the turn's inference, so a deferred strategy is
        # just the strategy it wraps.
        return [
            s.inner if isinstance(s, DeferredUserTurnStopStrategy) else s
            for s in strategies.stop or []
        ]

    async def _on_user_turn_inference_triggered(
        self, strategy: BaseUserTurnStopStrategy, speculation: UserTurnSpeculation | None
    ):
        # The controller runs the turn's inference itself once the stop
        # strategies have decided. A strategy only starts one to speculate.
        if not speculation:
            return

        await self._trigger_user_turn_inference_triggered(strategy, speculation)

    async def _on_user_turn_speculation_cancelled(self, strategy: BaseUserTurnStopStrategy):
        await self._call_event_handler("on_user_turn_speculation_cancelled")
        # Every strategy learns the speculation is gone, as when its service
        # withdraws it.
        await self.process_frame(UserTurnResponseCancelFrame())

    async def _on_user_turn_stopped(
        self, strategy: BaseUserTurnStopStrategy, params: UserTurnStoppedParams
    ):
        if self._stop_mode == StrategyMode.ALL:
            await self._add_user_turn_stop_decision(strategy, params)
        else:
            await self._first_user_turn_stop(strategy, params)

    async def _handle_user_turn_response_cancel(self):
        # Decisions from strategies that wait for the inference were made on
        # the cancelled answer, so they no longer hold. The other decisions
        # still say the user is done, including one that confirmed the answer,
        # so once they're all in, the turn's answer runs again.
        self._user_turn_inference = _UserTurnInference.NONE
        self._user_turn_stop_decisions = {
            strategy: replace(params, confirms_speculation=False)
            for strategy, params in self._user_turn_stop_decisions.items()
            if not strategy.waits_for_inference
        }
        if self._user_turn_stop_decisions and self._ready_for_inference():
            await self._trigger_user_turn_inference_triggered(
                next(iter(self._user_turn_stop_decisions))
            )

    async def _trigger_user_turn_inference_triggered(
        self,
        strategy: BaseUserTurnStopStrategy | None,
        speculation: UserTurnSpeculation | None = None,
    ):
        # Inference-triggered fires only while a turn is active. The turn
        # remains active afterward — only `on_user_turn_stopped` flips state.
        if not self._user_turn:
            return

        self._user_turn_inference = (
            _UserTurnInference.SPECULATIVE if speculation else _UserTurnInference.TRIGGERED
        )

        # Re-arm the stop watchdog so a stuck turn (inference fired but
        # finalization never arrives) still times out and finalizes.
        self._user_turn_stop_timeout_event.set()

        await self._call_event_handler("on_user_turn_inference_triggered", strategy, speculation)

    async def _first_user_turn_stop(
        self, strategy: BaseUserTurnStopStrategy, params: UserTurnStoppedParams
    ):
        if not self._user_turn:
            return

        # The first decision runs the turn's inference and ends the turn. A
        # stop strategy that waits for that inference still gets its say: the
        # turn then ends on its decision instead, keeping what the earlier
        # decisions said about the turn end.
        self._user_turn_stop_decisions[strategy] = params
        if not strategy.waits_for_inference and not self._user_turn_inference_triggered:
            await self._trigger_user_turn_inference_triggered(strategy)
        if self._user_turn_stop_decided():
            await self._trigger_user_turn_stop(strategy, self._user_turn_stop_params())

    async def _add_user_turn_stop_decision(
        self, strategy: BaseUserTurnStopStrategy, params: UserTurnStoppedParams
    ):
        # A decision counts while the user speaks, as when a service still
        # hears them while an eager answer runs. Only the turn end waits for
        # them to stop, and new speech clears the decisions.
        if not self._user_turn:
            return

        self._user_turn_stop_decisions[strategy] = params
        # The eager strategy confirms its speculation when the transcript
        # commits. If the speculation was already withdrawn, for example because
        # the LLM wrote ◐, there's nothing left to confirm, so it's handled like
        # the withdrawal: the decision becomes a plain one and the committed turn
        # gets its own inference.
        if (
            params.confirms_speculation
            and self._user_turn_inference != _UserTurnInference.SPECULATIVE
        ):
            await self._handle_user_turn_response_cancel()

        # Once every stop strategy that listens to the user has decided, run
        # the turn's inference. This comes before the turn-end check below, so
        # when this decision also ends the turn, the inference still starts
        # first.
        if not self._user_turn_inference_triggered and self._ready_for_inference():
            await self._trigger_user_turn_inference_triggered(strategy)

        if self._user_turn_stop_decided():
            await self._trigger_user_turn_stop(strategy, self._user_turn_stop_params())

    def _user_turn_stop_decided(self) -> bool:
        decided = self._user_turn_stop_decisions
        if self._stop_mode == StrategyMode.ALL:
            return len(decided) == len(self._user_turn_stop_strategies)
        # A FIRST list ends the turn on its first decision, or, when a stop
        # strategy waits for the inference, on that strategy's decision.
        waiting = [s for s in self._user_turn_stop_strategies if s.waits_for_inference]
        return any(s in decided for s in waiting) if waiting else bool(decided)

    async def _end_user_turn_if_decided(self):
        # Decisions made while the user was speaking end the turn once they
        # stop. New speech clears them, so they're about the speech just heard.
        if self._user_turn_stop_decisions and self._user_turn_stop_decided():
            strategy = next(reversed(self._user_turn_stop_decisions))
            await self._trigger_user_turn_stop(strategy, self._user_turn_stop_params())

    def _ready_for_inference(self) -> bool:
        # Only the stop strategies that wait for the inference are left to decide.
        return all(
            s in self._user_turn_stop_decisions
            for s in self._user_turn_stop_strategies
            if not s.waits_for_inference
        )

    def _user_turn_stop_params(self) -> UserTurnStoppedParams:
        # One decision saying the turn end was announced elsewhere, or that it
        # confirms a speculation, applies to the whole turn.
        decisions = self._user_turn_stop_decisions.values()
        return UserTurnStoppedParams(
            enable_user_speaking_frames=all(d.enable_user_speaking_frames for d in decisions),
            confirms_speculation=any(d.confirms_speculation for d in decisions),
        )

    async def _trigger_user_turn_stop(
        self, strategy: BaseUserTurnStopStrategy | None, params: UserTurnStoppedParams
    ):
        # Prevent two consecutive user turn stops.
        if not self._user_turn:
            return

        # Never end the turn while the user is audibly speaking. The decisions
        # stay, and the turn ends once the user stops.
        if self._user_speaking:
            return

        self._user_turn = False
        self._user_turn_stop_decisions.clear()
        self._user_turn_inference = _UserTurnInference.NONE
        self._user_turn_stop_timeout_event.set()

        # Notify every strategy that the turn has ended. Stop strategies reset
        # (and, e.g., drop a turn analyzer's buffered speech that must not
        # survive an externally-ended turn). Start strategies get the same
        # callback, but it's a no-op by default: their reset is turn-start
        # semantic, so resetting them here would be wrong.
        for s in self._user_turn_strategies.start or []:
            await s.handle_user_turn_stopped()
        for s in self._user_turn_stop_strategies:
            await s.handle_user_turn_stopped()

        await self._call_event_handler("on_user_turn_stopped", strategy, params)

    async def _user_turn_stop_timeout_task_handler(self):
        while True:
            try:
                await asyncio.wait_for(
                    self._user_turn_stop_timeout_event.wait(),
                    timeout=self._user_turn_stop_timeout,
                )
                self._user_turn_stop_timeout_event.clear()
            except TimeoutError:
                # A stop strategy with a decision of its own scheduled ends the
                # turn when it fires, so the timeout waits for it.
                decision_scheduled = any(
                    s.decision_scheduled for s in self._user_turn_stop_strategies
                )
                if self._user_turn and not self._user_speaking and not decision_scheduled:
                    await self._call_event_handler("on_user_turn_stop_timeout")
                    # The turn ends with what the decisions made so far said about
                    # it, so a confirmed speculation isn't answered a second time.
                    await self._trigger_user_turn_stop(None, self._user_turn_stop_params())
