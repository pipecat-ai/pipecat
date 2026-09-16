#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

import pytest

pytest.importorskip("typesafe_sdk")

from typesafe_sdk import Choice, Noul, TypeSafeAPITimeoutError

from pipecat.services.typesafe import TypeSafeJudge
from tests.typesafe_test_helpers import FakeTypeSafeClient, choice_response


class TestTypeSafeJudge(unittest.IsolatedAsyncioTestCase):
    async def test_ask_maps_answers(self):
        client = FakeTypeSafeClient(
            choice_response("yes", 0.9, {"yes": 0.9, "no": 0.1}, nouls={"asked": 0.2})
        )
        judge = TypeSafeJudge(client=client)  # type: ignore[arg-type]

        result = await judge.ask(
            {"user_reply": "yeah"},
            {
                "route": Choice(instructions="q", criteria={"yes": "y", "no": "n"}),
                "asked": Noul(instructions="did they ask?"),
            },
        )

        self.assertEqual(result.choices["route"].choice, "yes")
        self.assertEqual(result.choices["route"].probabilities, {"yes": 0.9, "no": 0.1})
        self.assertAlmostEqual(result.nouls["asked"].probability, 0.2)
        self.assertEqual(result.scores, {})
        self.assertEqual(result.model, "jev-test")
        self.assertEqual((result.input_tokens, result.output_tokens), (10, 2))
        self.assertIsNone(result.request_id)
        self.assertGreaterEqual(result.latency_secs, 0)

        state, questions = client.requests[0]
        self.assertEqual(state, {"user_reply": "yeah"})
        self.assertEqual(set(questions), {"route", "asked"})

    async def test_errors_propagate(self):
        client = FakeTypeSafeClient(error=TypeSafeAPITimeoutError("slow"))
        judge = TypeSafeJudge(client=client)  # type: ignore[arg-type]

        with self.assertRaises(TypeSafeAPITimeoutError):
            await judge.ask("x", {"q": Noul(instructions="?")})

    async def test_injected_client_is_not_closed(self):
        client = FakeTypeSafeClient(choice_response("yes", 1.0))
        judge = TypeSafeJudge(client=client)  # type: ignore[arg-type]

        judge.start()
        await judge.close()

        self.assertFalse(client.closed)

    async def test_warm_up_swallows_errors(self):
        client = FakeTypeSafeClient(error=RuntimeError("down"))
        judge = TypeSafeJudge(client=client)  # type: ignore[arg-type]

        await judge.warm_up()

        self.assertEqual(len(client.requests), 1)
