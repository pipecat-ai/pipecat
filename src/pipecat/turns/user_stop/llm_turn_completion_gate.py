#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Wrapper that holds a stop strategy's end of turn until the LLM says the turn is complete."""

from pipecat.frames.frames import (
    EagerEndOfTurnCancelFrame,
    Frame,
    LLMUpdateSettingsFrame,
    ProposedUserStartedSpeakingFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserTurnInferenceCompletedFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.settings import LLMSettings
from pipecat.turns.types import ProcessFrameResult, UserTurnSpeculation
from pipecat.turns.user_stop.base_user_turn_stop_strategy import (
    BaseUserTurnStopStrategy,
    UserTurnStoppedParams,
)
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig


class LLMTurnCompletionGate(BaseUserTurnStopStrategy):
    """Holds a stop strategy's end of turn until the LLM says the turn is complete.

    The LLM begins every response with a turn-completion marker: ● (complete),
    ◐ (incomplete short) or ○ (incomplete long). The wrapped strategy decides
    when to run inference and when the turn ends, as usual; the gate passes its
    inferences through and holds its end of turn until the response to the
    inference in flight starts with ●. A ◐ or ○ keeps the turn open for the
    user to go on.

    This works with any stop strategy, including one that answers a predicted
    end of turn speculatively, such as
    :class:`~pipecat.turns.user_stop.EagerUserTurnStopStrategy`: its
    confirmation of the speculation is held the same way, and a speculative
    response is only spoken once it carries the ●.

    While an end of turn is held, the wrapped strategy's repeated triggers for
    the same turn are dropped. The user going on releases it: a turn start
    releases any held end and withdraws a speculation it would confirm, and VAD
    hearing the user releases a held end that confirms none.

    On ``StartFrame``, the gate turns the markers on in the LLM with an
    ``LLMUpdateSettingsFrame`` that enables ``filter_incomplete_user_turns``.

    Use :func:`~pipecat.turns.user_turn_strategies.filter_incomplete` to gate
    every stop strategy in a set of strategies.
    """

    def __init__(
        self,
        inner: BaseUserTurnStopStrategy,
        *,
        config: UserTurnCompletionConfig | None = None,
        **kwargs,
    ):
        """Initialize the gate.

        Args:
            inner: The stop strategy whose end of turn is held.
            config: Configuration applied to the LLM via the
                ``filter_incomplete_user_turns`` setting on ``StartFrame``.
                Defaults to ``UserTurnCompletionConfig()``.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        super().__init__(**kwargs)
        self._inner = inner
        self._config = config or UserTurnCompletionConfig()
        # The wrapped strategy's end of turn, waiting for the LLM's ●.
        self._held: UserTurnStoppedParams | None = None
        # Whether the LLM said ● for the inference in flight.
        self._complete = False

        inner.add_event_handler("on_push_frame", self._on_push_frame)
        inner.add_event_handler("on_broadcast_frame", self._on_broadcast_frame)
        inner.add_event_handler("on_user_turn_inference_triggered", self._on_inference_triggered)
        inner.add_event_handler(
            "on_user_turn_speculation_cancelled", self._on_speculation_cancelled
        )
        inner.add_event_handler("on_user_turn_stopped", self._on_user_turn_stopped)

    @property
    def inner(self) -> BaseUserTurnStopStrategy:
        """Return the wrapped strategy."""
        return self._inner

    @property
    def config(self) -> UserTurnCompletionConfig:
        """Return the configured ``UserTurnCompletionConfig``."""
        return self._config

    @property
    def resolves_proposed_turn_stop_frames(self) -> bool:
        """Report what the wrapped strategy does with proposals."""
        return self._inner.resolves_proposed_turn_stop_frames

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the gate and the wrapped strategy."""
        await super().setup(setup)
        await self._inner.setup(setup)

    async def cleanup(self):
        """Clean up the gate and the wrapped strategy."""
        await super().cleanup()
        await self._inner.cleanup()

    async def handle_user_turn_started(self):
        """Forget the previous turn, and forward the turn start."""
        self._reset()
        await self._inner.handle_user_turn_started()

    async def handle_user_turn_stopped(self):
        """Withdraw a speculation the turn ended without, and forward the turn stop."""
        if self._held and self._held.confirms_speculation:
            # The turn ended some other way, such as the stop timeout, so the
            # response the held end would confirm is never spoken.
            await self.trigger_user_turn_speculation_cancelled()
        self._reset()
        await self._inner.handle_user_turn_stopped()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult | None:
        """Pass the frame to the wrapped strategy, and follow the LLM and the user.

        Args:
            frame: The frame to be analyzed.

        Returns:
            What the wrapped strategy returned.
        """
        if isinstance(frame, StartFrame):
            await self._configure_llm()

        result = await self._inner.process_frame(frame)

        if isinstance(frame, UserTurnInferenceCompletedFrame):
            await self._complete_turn()
        elif isinstance(frame, (ProposedUserStartedSpeakingFrame, UserStartedSpeakingFrame)):
            await self._release(withdraw_speculation=True)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._release(withdraw_speculation=False)
        elif isinstance(frame, EagerEndOfTurnCancelFrame):
            self._complete = False

        return result

    def _reset(self):
        self._held = None
        self._complete = False

    async def _configure_llm(self):
        await self.push_frame(
            LLMUpdateSettingsFrame(
                delta=LLMSettings(
                    filter_incomplete_user_turns=True,
                    user_turn_completion_config=self._config,
                ),
                reach_inactive_services=True,
            )
        )

    async def _on_push_frame(
        self,
        inner: BaseUserTurnStopStrategy,
        frame: Frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ):
        await self.push_frame(frame, direction)

    async def _on_broadcast_frame(
        self, inner: BaseUserTurnStopStrategy, frame_cls: type[Frame], **kwargs
    ):
        await self.broadcast_frame(frame_cls, **kwargs)

    async def _on_inference_triggered(
        self, inner: BaseUserTurnStopStrategy, speculation: UserTurnSpeculation | None
    ):
        """Pass an inference through, unless it re-checks a turn waiting for ●."""
        if self._held:
            return
        self._complete = False
        await self.trigger_user_turn_inference_triggered(speculation=speculation)

    async def _on_speculation_cancelled(self, inner: BaseUserTurnStopStrategy):
        self._complete = False
        await self.trigger_user_turn_speculation_cancelled()

    async def _on_user_turn_stopped(
        self, inner: BaseUserTurnStopStrategy, params: UserTurnStoppedParams
    ):
        """End the turn if the LLM already said ●, or hold the end until it does."""
        if self._held:
            return
        if self._complete:
            await self._finalize(params)
        else:
            self._held = params

    async def _complete_turn(self):
        """Note the LLM's ●, ending a turn whose end is held."""
        self._complete = True
        if self._held:
            params, self._held = self._held, None
            await self._finalize(params)

    async def _release(self, *, withdraw_speculation: bool):
        """Drop a held end of turn, since the user went on."""
        params = self._held
        if params is None or (params.confirms_speculation and not withdraw_speculation):
            return
        self._held = None
        if params.confirms_speculation:
            # The response it would confirm was never spoken.
            await self.trigger_user_turn_speculation_cancelled()

    async def _finalize(self, params: UserTurnStoppedParams):
        await self.trigger_user_turn_finalized(
            enable_user_speaking_frames=params.enable_user_speaking_frames,
            confirms_speculation=params.confirms_speculation,
        )
