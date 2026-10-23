#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module defines a controller for managing user turn lifecycle."""

import asyncio
from dataclasses import replace

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
from pipecat.turns.types import ProcessFrameResult, UserTurnSpeculation
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


class UserTurnController(BaseObject):
    """Controller for managing user turn lifecycle.

    This class manages user turn state (active/inactive), handles start and stop
    strategies, and emits events when user turns begin, end, or timeout occurs.

    How a turn starts and ends follows the ``start_mode`` and ``stop_mode`` of
    its :class:`~pipecat.turns.user_turn_strategies.UserTurnStrategiesParams`.
    With ``StrategyMode.FIRST`` the first strategy to decide starts or ends
    the turn. With ``StrategyMode.ALL`` each decision is a vote, and the turn
    starts or ends once every strategy in the list has voted. Turns start on
    the first start strategy and end once every stop strategy agrees by
    default. A cancelled answer voids the stop votes cast on it.

    In an ``ALL`` list the controller starts the turn's inference itself,
    once every stop strategy voting on the user has voted. A stop strategy
    that votes on the inference instead, such as one reading the LLM's
    turn-completion marker, then votes on that inference. An eager end of
    turn still starts its speculation at once.

    Event handlers available:

    - on_user_turn_started: Emitted when a user turn starts.
    - on_user_turn_inference_triggered: Emitted when enough signal exists to
      start LLM inference, carrying a `UserTurnSpeculation` when the turn it
      answers hasn't ended yet. Fires together with `on_user_turn_stopped` for
      most strategies; fires alone when the turn's end waits for a vote on
      the inference, or for an eager end of turn to be confirmed.
    - on_user_turn_stopped: Emitted when a user turn is semantically final.
    - on_user_turn_answer_cancelled: Emitted when the user speaking again
      cancels the answer started for the open turn.
    - on_user_turn_stop_timeout: Emitted if no stop strategy triggers before timeout.
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

        @controller.event_handler("on_user_turn_stop_timeout")
        async def on_user_turn_stop_timeout(controller):
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
            user_turn_stop_timeout: Timeout in seconds to automatically stop a user turn
                if no activity is detected.
        """
        super().__init__()

        self._user_turn_strategies = user_turn_strategies
        self._user_turn_strategies_params = (
            user_turn_strategies_params or UserTurnStrategiesParams()
        )
        self._user_turn_stop_timeout = user_turn_stop_timeout
        self._setup: FrameProcessorSetup | None = None

        self._user_speaking = False

        self._user_turn = False
        self._user_turn_start_votes: dict[BaseUserTurnStartStrategy, UserTurnStartedParams] = {}
        self._user_turn_stop_votes: dict[BaseUserTurnStopStrategy, UserTurnStoppedParams] = {}
        # Whether the turn's answer started for the stop votes cast so far.
        self._user_turn_answered = False
        # Whether the turn's answer in flight is a speculation, so a vote
        # confirming one that was withdrawn meanwhile is told apart.
        self._user_turn_speculating = False
        self._user_turn_stop_timeout_event = asyncio.Event()
        self._user_turn_stop_timeout_task: asyncio.Task | None = None

        self._register_event_handler("on_push_frame", sync=True)
        self._register_event_handler("on_broadcast_frame", sync=True)
        self._register_event_handler("on_user_turn_started", sync=True)
        self._register_event_handler("on_user_turn_inference_triggered", sync=True)
        self._register_event_handler("on_user_turn_speculation_cancelled", sync=True)
        self._register_event_handler("on_user_turn_stopped", sync=True)
        self._register_event_handler("on_user_turn_answer_cancelled", sync=True)
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
        self._user_turn_start_votes.clear()
        self._user_turn_stop_votes.clear()
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
        return any(
            s.resolves_proposed_turn_stop_frames for s in self._user_turn_strategies.stop or []
        )

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

        for strategy in self._user_turn_strategies.start or []:
            result = await strategy.process_frame(frame)
            # A strategy that votes needs every frame to decide on.
            if result == ProcessFrameResult.STOP and not self._start_strategies_vote:
                break

        for strategy in self._user_turn_strategies.stop or []:
            result = await strategy.process_frame(frame)
            # A strategy that votes needs every frame to decide on.
            if result == ProcessFrameResult.STOP and not self._stop_strategies_vote:
                break

        # Handled once the stop strategies have seen it, so an inference started
        # here is the one they then wait for.
        if isinstance(frame, UserTurnResponseCancelFrame):
            await self._handle_user_turn_response_cancel()

    @property
    def _start_strategies_vote(self) -> bool:
        return self._user_turn_strategies_params.start_mode == StrategyMode.ALL

    @property
    def _stop_strategies_vote(self) -> bool:
        return self._user_turn_strategies_params.stop_mode == StrategyMode.ALL

    async def _setup_strategies(self):
        if not self._setup:
            raise RuntimeError(f"{self} was not properly set up")

        for s in self._user_turn_strategies.start or []:
            await s.setup(self._setup)
            s.add_event_handler("on_push_frame", self._on_push_frame)
            s.add_event_handler("on_broadcast_frame", self._on_broadcast_frame)
            s.add_event_handler("on_user_turn_started", self._on_user_turn_started)
            s.add_event_handler("on_reset_aggregation", self._on_reset_aggregation)

        for s in self._user_turn_strategies.stop or []:
            await s.setup(self._setup)
            s.add_event_handler("on_push_frame", self._on_push_frame)
            s.add_event_handler("on_broadcast_frame", self._on_broadcast_frame)
            s.add_event_handler(
                "on_user_turn_inference_triggered", self._on_user_turn_inference_triggered
            )
            s.add_event_handler(
                "on_user_turn_speculation_cancelled", self._on_user_turn_speculation_cancelled
            )
            self._user_turn_stop_source(s).add_event_handler(
                "on_user_turn_stopped", self._on_user_turn_stopped
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

        for s in self._user_turn_strategies.stop or []:
            await s.cleanup()
            s.remove_event_handler("on_push_frame", self._on_push_frame)
            s.remove_event_handler("on_broadcast_frame", self._on_broadcast_frame)
            s.remove_event_handler(
                "on_user_turn_inference_triggered", self._on_user_turn_inference_triggered
            )
            s.remove_event_handler(
                "on_user_turn_speculation_cancelled", self._on_user_turn_speculation_cancelled
            )
            self._user_turn_stop_source(s).remove_event_handler(
                "on_user_turn_stopped", self._on_user_turn_stopped
            )

    def _user_turn_stop_source(
        self, strategy: BaseUserTurnStopStrategy
    ) -> BaseUserTurnStopStrategy:
        # Every stop strategy votes, so a deferred one votes as the strategy
        # it wraps.
        if self._stop_strategies_vote and isinstance(strategy, DeferredUserTurnStopStrategy):
            return strategy.inner
        return strategy

    async def _handle_user_started_speaking(
        self, frame: UserStartedSpeakingFrame | ProposedUserStartedSpeakingFrame
    ):
        self._user_speaking = True
        await self._clear_user_turn_stop_votes()

        # The user started talking, let's reset the user turn timeout.
        self._user_turn_stop_timeout_event.set()

    async def _handle_user_stopped_speaking(
        self, frame: UserStoppedSpeakingFrame | ProposedUserStoppedSpeakingFrame
    ):
        self._user_speaking = False

        # The user stopped talking, let's reset the user turn timeout.
        self._user_turn_stop_timeout_event.set()

    async def _handle_vad_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        self._user_speaking = True
        await self._clear_user_turn_stop_votes()

        # The user started talking, let's reset the user turn timeout.
        self._user_turn_stop_timeout_event.set()

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        self._user_speaking = False

        # The user stopped talking, let's reset the user turn timeout.
        self._user_turn_stop_timeout_event.set()

    async def _handle_user_turn_response_cancel(self):
        # The votes cast on the cancelled answer no longer hold. A vote that
        # confirmed it still says the user is done, so once every vote on the
        # user is in, the turn's answer runs again.
        self._user_turn_speculating = False
        self._user_turn_answered = False
        self._user_turn_stop_votes = {
            strategy: replace(params, confirms_speculation=False)
            for strategy, params in self._user_turn_stop_votes.items()
            if params.confirms_speculation
        }
        if self._user_turn_stop_votes and self._every_user_vote_in():
            await self._trigger_user_turn_inference_triggered(
                next(iter(self._user_turn_stop_votes))
            )

    async def _clear_user_turn_stop_votes(self):
        # Speech in an open turn voids the votes cast so far, and the answer
        # started for them.
        answered = self._user_turn_answered
        self._user_turn_stop_votes.clear()
        self._user_turn_answered = False
        self._user_turn_speculating = False
        if answered:
            await self._call_event_handler("on_user_turn_answer_cancelled")

    async def _handle_transcription(self, frame: TranscriptionFrame | InterimTranscriptionFrame):
        # We have received a transcription, let's reset the user turn timeout.
        self._user_turn_stop_timeout_event.set()

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

    async def _on_user_turn_speculation_cancelled(self, strategy: BaseUserTurnStopStrategy):
        await self._call_event_handler("on_user_turn_speculation_cancelled")
        # Every strategy learns the speculation is gone, as when its service
        # withdraws it.
        await self.process_frame(UserTurnResponseCancelFrame())

    async def _on_user_turn_started(
        self,
        strategy: BaseUserTurnStartStrategy,
        params: UserTurnStartedParams,
    ):
        if self._start_strategies_vote:
            await self._vote_user_turn_start(strategy, params)
        else:
            await self._trigger_user_turn_start(strategy, params)

    async def _on_user_turn_inference_triggered(
        self, strategy: BaseUserTurnStopStrategy, speculation: UserTurnSpeculation | None
    ):
        # In a voting list, a stop strategy's own trigger only counts as its
        # vote: the controller runs the turn's inference itself. A speculation
        # runs at once.
        if self._stop_strategies_vote and not speculation:
            return

        await self._trigger_user_turn_inference_triggered(strategy, speculation)

    async def _on_user_turn_stopped(
        self, strategy: BaseUserTurnStopStrategy, params: UserTurnStoppedParams
    ):
        if self._stop_strategies_vote:
            await self._vote_user_turn_stop(strategy, params)
        else:
            await self._trigger_user_turn_stop(strategy, params)

    async def _on_reset_aggregation(self, strategy: BaseUserTurnStartStrategy):
        await self._call_event_handler("on_reset_aggregation", strategy)

    async def _vote_user_turn_start(
        self, strategy: BaseUserTurnStartStrategy, params: UserTurnStartedParams
    ):
        if self._user_turn:
            return

        self._user_turn_start_votes[strategy] = params

        if len(self._user_turn_start_votes) == len(self._user_turn_strategies.start or []):
            await self._trigger_user_turn_start(strategy, self._user_turn_start_params())

    def _user_turn_start_params(self) -> UserTurnStartedParams:
        # One vote saying the turn shouldn't interrupt, or was announced
        # elsewhere, applies to the whole turn.
        votes = self._user_turn_start_votes.values()
        return UserTurnStartedParams(
            enable_interruptions=all(v.enable_interruptions for v in votes),
            enable_user_speaking_frames=all(v.enable_user_speaking_frames for v in votes),
        )

    async def _trigger_user_turn_start(
        self, strategy: BaseUserTurnStartStrategy | None, params: UserTurnStartedParams
    ):
        # Prevent two consecutive user turn starts.
        if self._user_turn:
            return

        self._user_turn = True
        self._user_turn_start_votes.clear()
        self._user_turn_answered = False
        self._user_turn_speculating = False
        self._user_turn_stop_timeout_event.set()

        # Notify every strategy that the turn has started. Start strategies
        # ready themselves for the next detection; stop strategies arm to detect
        # this turn's end. A strategy resets whatever per-turn state it keeps
        # inside its own handle_user_turn_started.
        for s in self._user_turn_strategies.start or []:
            await s.handle_user_turn_started()
        for s in self._user_turn_strategies.stop or []:
            await s.handle_user_turn_started()

        await self._call_event_handler("on_user_turn_started", strategy, params)

    async def _trigger_user_turn_inference_triggered(
        self,
        strategy: BaseUserTurnStopStrategy | None,
        speculation: UserTurnSpeculation | None = None,
    ):
        # Inference-triggered fires only while a turn is active. The turn
        # remains active afterward — only `on_user_turn_stopped` flips state.
        if not self._user_turn:
            return

        self._user_turn_answered = True

        # Re-arm the stop watchdog so a stuck turn (inference fired but
        # finalization never arrives) still times out and finalizes.
        self._user_turn_stop_timeout_event.set()

        self._user_turn_speculating = speculation is not None
        await self._call_event_handler("on_user_turn_inference_triggered", strategy, speculation)

    async def _vote_user_turn_stop(
        self, strategy: BaseUserTurnStopStrategy, params: UserTurnStoppedParams
    ):
        # A vote counts while the user speaks, as when a service still hears
        # them while an eager answer runs. Only the turn end waits
        # for them to stop, and new speech clears the votes.
        if not self._user_turn:
            return

        self._user_turn_stop_votes[strategy] = params
        # A confirmation that crossed the withdrawal of its speculation
        # confirms nothing, the same as one cast before it.
        if params.confirms_speculation and not self._user_turn_speculating:
            await self._handle_user_turn_response_cancel()

        # Started before the turn can end, so the last vote on the user starts
        # the inference ahead of the turn end it may also bring.
        if not self._user_turn_answered and self._every_user_vote_in():
            await self._trigger_user_turn_inference_triggered(strategy)

        if len(self._user_turn_stop_votes) == len(self._user_turn_strategies.stop or []):
            await self._trigger_user_turn_stop(strategy, self._user_turn_stop_params())

    def _every_user_vote_in(self) -> bool:
        # Only the stop strategies voting on the inference are left to vote.
        return all(
            self._user_turn_stop_source(s) in self._user_turn_stop_votes
            for s in self._user_turn_strategies.stop or []
            if not s.votes_on_inference
        )

    def _user_turn_stop_params(self) -> UserTurnStoppedParams:
        # One vote saying the turn end was announced elsewhere, or that it
        # confirms a speculation, applies to the whole turn.
        votes = self._user_turn_stop_votes.values()
        return UserTurnStoppedParams(
            enable_user_speaking_frames=all(v.enable_user_speaking_frames for v in votes),
            confirms_speculation=any(v.confirms_speculation for v in votes),
        )

    async def _trigger_user_turn_stop(
        self, strategy: BaseUserTurnStopStrategy | None, params: UserTurnStoppedParams
    ):
        # Prevent two consecutive user turn stops.
        if not self._user_turn:
            return

        # Never finalize while the user is audibly speaking. A stop strategy can
        # finalize on a latent signal (e.g. an LLM ● that resolves after the
        # user resumed), which is stale by the time it arrives. Keep the turn
        # open so the next inference re-evaluates; the watchdog still finalizes
        # if the user then falls silent. Detector strategies only finalize once
        # the user has stopped, so this is a no-op for them.
        if self._user_speaking:
            return

        self._user_turn = False
        self._user_turn_start_votes.clear()
        self._user_turn_stop_votes.clear()
        self._user_turn_answered = False
        self._user_turn_speculating = False
        self._user_turn_stop_timeout_event.set()

        # Notify every strategy that the turn has ended. Stop strategies reset
        # (and, e.g., drop a turn analyzer's buffered speech that must not
        # survive an externally-ended turn). Start strategies get the same
        # callback, but it's a no-op by default: their reset is turn-start
        # semantic, so resetting them here would be wrong.
        for s in self._user_turn_strategies.start or []:
            await s.handle_user_turn_stopped()
        for s in self._user_turn_strategies.stop or []:
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
                if self._user_turn and not self._user_speaking:
                    await self._call_event_handler("on_user_turn_stop_timeout")
                    await self._trigger_user_turn_stop(
                        None, UserTurnStoppedParams(enable_user_speaking_frames=True)
                    )
