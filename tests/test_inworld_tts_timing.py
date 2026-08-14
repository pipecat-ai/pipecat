#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression tests for per-context generation timing in InworldTTSService.

Word timestamps and the flushCompleted offset must stay scoped to their
contextId: a late timestamp or flush event from an interrupted context must
not shift a newer context's word PTS.
"""

import json
import unittest

from pipecat.services.inworld.tts import InworldTTSService


def _ts(words, starts, ends):
    return {
        "wordAlignment": {
            "words": words,
            "wordStartTimeSeconds": starts,
            "wordEndTimeSeconds": ends,
        }
    }


class TestInworldContextTiming(unittest.IsolatedAsyncioTestCase):
    def _service(self) -> InworldTTSService:
        return InworldTTSService(api_key="test-key")

    def test_late_event_from_interrupted_context_does_not_shift_new_context(self):
        """A late timestamp + flush for A must not move B's word timestamps."""
        svc = self._service()
        svc._register_context_timing("A")
        svc._register_context_timing("B")

        svc._calculate_word_times(_ts(["hello"], [0.25], [10.75]), "A")
        svc._handle_flush_completed("A")  # A's cumulative time now 10.75

        times_b = svc._calculate_word_times(_ts(["world"], [0.25], [1.0]), "B")
        assert times_b == [("world", 0.25)]

    def test_generations_accumulate_within_one_context(self):
        """Multiple generations keep accumulating monotonically inside one context."""
        svc = self._service()
        svc._register_context_timing("B")

        svc._calculate_word_times(_ts(["one", "two"], [0.25, 0.5], [0.6, 1.0]), "B")
        svc._handle_flush_completed("B")  # cumulative(B) = 1.0

        times = svc._calculate_word_times(_ts(["three"], [0.5], [0.9]), "B")
        assert times == [("three", 1.5)]

    def test_late_events_for_closed_context_are_dropped(self):
        """Events for an interrupted/closed context are ignored, not replayed."""
        svc = self._service()
        svc._register_context_timing("A")
        svc._calculate_word_times(_ts(["hello"], [0.25], [0.9]), "A")

        svc._discard_context_timing("A")
        assert svc._calculate_word_times(_ts(["late"], [0.1], [0.5]), "A") == []
        svc._handle_flush_completed("A")  # must not raise or recreate state
        assert "A" not in svc._timing_by_context_id

    def test_flush_only_advances_its_own_context(self):
        svc = self._service()
        svc._register_context_timing("A")
        svc._register_context_timing("B")
        svc._calculate_word_times(_ts(["first"], [0.0], [5.0]), "A")
        svc._calculate_word_times(_ts(["second"], [0.0], [2.0]), "B")

        svc._handle_flush_completed("A")
        assert svc._timing_by_context_id["A"].cumulative_time_s == 5.0
        assert svc._timing_by_context_id["B"].cumulative_time_s == 0.0

    def test_recreated_context_starts_fresh(self):
        """A context recreated after interruption starts from a zero offset."""
        svc = self._service()
        svc._register_context_timing("A")
        svc._calculate_word_times(_ts(["old"], [0.0], [9.0]), "A")
        svc._discard_context_timing("A")

        svc._register_context_timing("A")  # new turn, same id
        times = svc._calculate_word_times(_ts(["new"], [0.25], [1.0]), "A")
        assert times == [("new", 0.25)]

    async def test_receive_loop_scopes_timing_by_context(self):
        """The receive loop routes timestamps and flushes per contextId."""
        from unittest.mock import patch

        svc = self._service()
        svc._register_context_timing("A")
        svc._register_context_timing("B")

        messages = [
            json.dumps(
                {
                    "result": {
                        "audioChunk": {"timestampInfo": _ts(["hello"], [0.25], [10.75])},
                        "contextId": "A",
                    }
                }
            ),
            json.dumps({"result": {"flushCompleted": {}, "contextId": "A"}}),
            json.dumps(
                {
                    "result": {
                        "audioChunk": {"timestampInfo": _ts(["world"], [0.25], [1.0])},
                        "contextId": "B",
                    }
                }
            ),
        ]

        async def _fake_ws():
            for message in messages:
                yield message

        captured = []

        async def _capture(word_times, ctx_id, pre_merge_tokens):
            captured.append((word_times, ctx_id))

        with (
            patch.object(svc, "_get_websocket", return_value=_fake_ws()),
            patch.object(svc, "add_word_timestamps", side_effect=_capture),
        ):
            await svc._receive_messages()

        assert captured == [([("hello", 0.25)], "A"), ([("world", 0.25)], "B")]
        # A's flush advanced only A's offset
        assert svc._timing_by_context_id["A"].cumulative_time_s == 10.75
        assert svc._timing_by_context_id["B"].cumulative_time_s == 0.0
