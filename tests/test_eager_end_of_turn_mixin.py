#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    EagerEndOfTurnCancelFrame,
    EagerEndOfTurnTranscriptionFrame,
    Frame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.eager_end_of_turn_mixin import EagerEndOfTurnSTTServiceMixin


class Predictor(EagerEndOfTurnSTTServiceMixin, FrameProcessor):
    """Stands in for an STT service that predicts the end of a turn."""

    def __init__(self):
        super().__init__()
        self.pushed: list[Frame] = []

    async def push_frame(self, frame: Frame, direction=FrameDirection.DOWNSTREAM):
        self.pushed.append(frame)


class TestEagerEndOfTurnMixin(unittest.IsolatedAsyncioTestCase):
    async def test_a_withdrawal_names_the_prediction_it_withdraws(self):
        service = Predictor()

        await service._push_eager_end_of_turn("book a flight", user_id="user")
        await service._cancel_eager_end_of_turn()

        prediction, withdrawal = service.pushed
        assert isinstance(prediction, EagerEndOfTurnTranscriptionFrame)
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
