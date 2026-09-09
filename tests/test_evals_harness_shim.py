#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The deprecated ``pipecat.evals.harness`` path still exports what it used to."""

import unittest
import warnings

from pipecat.evals import client, results, script, script_driver, script_session, session


class TestHarnessShim(unittest.TestCase):
    def test_every_former_export_resolves_to_its_new_home(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from pipecat.evals import harness

        expected = {
            "EvalSession": session.EvalSession,
            "DEFAULT_EVENT_TIMEOUT_MS": script_session.DEFAULT_EVENT_TIMEOUT_MS,
            "BOT_READY_TIMEOUT_S": client.BOT_READY_TIMEOUT_S,
            "SEND_AFTER_MAX_WAIT_S": script_driver.SEND_AFTER_MAX_WAIT_S,
            "SEND_AFTER_POLL_S": script_driver.SEND_AFTER_POLL_S,
            "FAILURE_KINDS": results.FAILURE_KINDS,
            "TURN_STATUSES": results.TURN_STATUSES,
            "EvalAssertionFailure": results.EvalAssertionFailure,
            "EvalResult": results.EvalResult,
            "EvalTurnResult": results.EvalTurnResult,
            "EvalTurnProgress": results.EvalTurnProgress,
            "EvalScenario": script.EvalScenario,
            "EvalTurn": script.EvalTurn,
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertIs(getattr(harness, name), value)
