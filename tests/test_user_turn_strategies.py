#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.turns.user_start import (
    ExternalUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import (
    ExternalUserTurnStopStrategy,
    LLMTurnCompletionUserTurnStopStrategy,
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import (
    ExternalUserTurnStrategies,
    FilterIncompleteUserTurnStrategies,
    MergeMode,
    UserTurnStrategies,
    UserTurnStrategiesParams,
)

EXTEND_STOP = UserTurnStrategiesParams(stop_merge_mode=MergeMode.EXTEND)


class TestMergeWithRecommendation(unittest.TestCase):
    """A bot's user turn strategies merge with a service's, list by list."""

    def test_a_bots_lists_replace_the_services_by_default(self):
        stop = SpeechTimeoutUserTurnStopStrategy()
        strategies = UserTurnStrategies(stop=[stop])

        merged = strategies.merge_with(ExternalUserTurnStrategies(), UserTurnStrategiesParams())

        self.assertEqual(merged.stop, [stop])
        self.assertEqual(merged.start, strategies.start)

    def test_an_extending_list_joins_the_services(self):
        stop = SpeechTimeoutUserTurnStopStrategy()
        recommended = ExternalUserTurnStrategies()

        merged = UserTurnStrategies(stop=[stop]).merge_with(recommended, EXTEND_STOP)

        self.assertEqual(merged.stop, [*(recommended.stop or []), stop])

    def test_extending_a_list_left_out_keeps_the_services(self):
        recommended = ExternalUserTurnStrategies()
        params = UserTurnStrategiesParams(start_merge_mode=MergeMode.EXTEND)

        merged = UserTurnStrategies(stop=[SpeechTimeoutUserTurnStopStrategy()]).merge_with(
            recommended, params
        )

        self.assertEqual(merged.start, recommended.start)

    def test_without_a_recommendation_an_extending_list_joins_the_defaults(self):
        start = ExternalUserTurnStartStrategy()
        strategies = UserTurnStrategies(start=[start], stop=[SpeechTimeoutUserTurnStopStrategy()])
        params = UserTurnStrategiesParams(start_merge_mode=MergeMode.EXTEND)

        merged = strategies.merge_with(None, params)

        self.assertEqual(
            [type(s) for s in merged.start or []],
            [
                VADUserTurnStartStrategy,
                TranscriptionUserTurnStartStrategy,
                ExternalUserTurnStartStrategy,
            ],
        )
        self.assertIs((merged.start or [])[-1], start)

    def test_only_two_replacing_lists_leave_a_recommendation_unused(self):
        self.assertTrue(UserTurnStrategiesParams().replaces_service_strategies)
        self.assertFalse(EXTEND_STOP.replaces_service_strategies)
        self.assertFalse(
            UserTurnStrategiesParams(start_merge_mode=MergeMode.EXTEND).replaces_service_strategies
        )


class TestFilterIncompleteMerge(unittest.TestCase):
    """Filtering incomplete turns replaces a service's stop strategies, or joins them."""

    def test_replacing_filters_with_the_default_stop_strategy(self):
        merged = FilterIncompleteUserTurnStrategies().merge_with(
            ExternalUserTurnStrategies(), UserTurnStrategiesParams()
        )

        self.assertEqual(
            [type(s) for s in merged.stop or []],
            [TurnAnalyzerUserTurnStopStrategy, LLMTurnCompletionUserTurnStopStrategy],
        )

    def test_extending_adds_only_the_llm_gate(self):
        merged = FilterIncompleteUserTurnStrategies().merge_with(
            ExternalUserTurnStrategies(), EXTEND_STOP
        )

        self.assertEqual(
            [type(s) for s in merged.stop or []],
            [ExternalUserTurnStopStrategy, LLMTurnCompletionUserTurnStopStrategy],
        )

    def test_extending_adds_the_stop_strategies_given_with_the_llm_gate(self):
        strategies = FilterIncompleteUserTurnStrategies(stop=[SpeechTimeoutUserTurnStopStrategy()])

        merged = strategies.merge_with(ExternalUserTurnStrategies(), EXTEND_STOP)

        self.assertEqual(
            [type(s) for s in merged.stop or []],
            [
                ExternalUserTurnStopStrategy,
                SpeechTimeoutUserTurnStopStrategy,
                LLMTurnCompletionUserTurnStopStrategy,
            ],
        )


if __name__ == "__main__":
    unittest.main()
