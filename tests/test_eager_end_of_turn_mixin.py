#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    EagerEndOfTurnCancelFrame,
    EagerTranscriptionFrame,
    Frame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.eager_end_of_turn_mixin import EagerEndOfTurnSTTServiceMixin
from pipecat.turns.user_stop import EagerUserTurnStopStrategy, ExactMatch
from pipecat.turns.user_turn_strategies import (
    EagerUserTurnStrategies,
    ExternalUserTurnStrategies,
)


class Predictor(EagerEndOfTurnSTTServiceMixin, FrameProcessor):
    """Stands in for an STT service that predicts the end of a turn."""

    def __init__(self, *, enable_eager_end_of_turn: bool = True, **kwargs):
        super().__init__(enable_eager_end_of_turn=enable_eager_end_of_turn, **kwargs)
        self.pushed: list[Frame] = []

    async def push_frame(self, frame: Frame, direction=FrameDirection.DOWNSTREAM):
        self.pushed.append(frame)


class TestEagerEndOfTurnMixin(unittest.IsolatedAsyncioTestCase):
    async def test_a_withdrawal_names_the_prediction_it_withdraws(self):
        service = Predictor()

        await service._push_eager_end_of_turn("book a flight", user_id="user")
        await service._cancel_eager_end_of_turn()

        prediction, withdrawal = service.pushed
        assert isinstance(prediction, EagerTranscriptionFrame)
        assert prediction.text == "book a flight"
        assert prediction.user_id == "user"
        assert isinstance(withdrawal, EagerEndOfTurnCancelFrame)
        assert withdrawal.speculation_id == prediction.speculation_id
        assert service.eager_speculation_id is None

    async def test_each_prediction_gets_its_own_id(self):
        service = Predictor()

        await service._push_eager_end_of_turn("i think", user_id="user")
        await service._push_eager_end_of_turn("i think i'll book it", user_id="user")

        first, second = service.pushed
        assert first.speculation_id and second.speculation_id
        assert first.speculation_id != second.speculation_id

    async def test_a_committed_turn_resolves_the_prediction_without_withdrawing_it(self):
        service = Predictor()

        await service._push_eager_end_of_turn("book a flight", user_id="user")
        service._clear_eager_end_of_turn()
        # A committed turn settles the prediction on its own; withdrawing it
        # afterwards would discard a response the committed transcript may keep.
        await service._cancel_eager_end_of_turn()

        assert not any(isinstance(f, EagerEndOfTurnCancelFrame) for f in service.pushed)
        assert service.eager_speculation_id is None

    async def test_withdrawing_without_a_prediction_does_nothing(self):
        service = Predictor()

        await service._cancel_eager_end_of_turn()
        service._clear_eager_end_of_turn()

        assert service.pushed == []
        assert service.eager_speculation_id is None


class TestEagerEndOfTurnIsOptIn(unittest.IsolatedAsyncioTestCase):
    async def test_predictions_are_not_reported_while_it_is_off(self):
        # A service reports what its protocol gives it either way; the mixin is
        # what decides whether the prediction goes anywhere.
        service = Predictor(enable_eager_end_of_turn=False)

        await service._push_eager_end_of_turn("book a flight", user_id="user")
        await service._cancel_eager_end_of_turn()
        service._clear_eager_end_of_turn()

        assert service.pushed == []
        assert service.eager_speculation_id is None
        assert not service.eager_end_of_turn_enabled

    def test_it_recommends_eager_strategies_only_when_enabled(self):
        assert isinstance(
            Predictor().recommended_user_turn_strategies(enable_interruptions=True),
            EagerUserTurnStrategies,
        )

        recommended = Predictor(enable_eager_end_of_turn=False).recommended_user_turn_strategies(
            enable_interruptions=True
        )
        assert isinstance(recommended, ExternalUserTurnStrategies)
        assert not isinstance(recommended, EagerUserTurnStrategies)

    def test_the_recommendation_carries_the_match_policy_and_interruptions(self):
        service = Predictor(eager_match_policy=ExactMatch())

        recommended = service.recommended_user_turn_strategies(enable_interruptions=False)

        assert isinstance(recommended.stop[0], EagerUserTurnStopStrategy)
        assert isinstance(recommended.stop[0].match_policy, ExactMatch)
        assert recommended.enable_interruptions is False
