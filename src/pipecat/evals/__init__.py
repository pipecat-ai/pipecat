"""Behavioral harness for Pipecat agents.

This package provides a scenario-based test runner that drives a bot via the
eval transport and asserts on the semantic event stream it emits. See
:mod:`pipecat.evals.scenario` for the YAML schema and
:mod:`pipecat.evals.harness` for the runner.
"""

from pipecat.evals.harness import (
    AssertionFailure,
    ScenarioResult,
    run_scenario,
)
from pipecat.evals.scenario import (
    Expectation,
    Scenario,
    SendAfter,
    Turn,
    load_scenario,
)

__all__ = [
    "AssertionFailure",
    "Expectation",
    "Scenario",
    "ScenarioResult",
    "SendAfter",
    "Turn",
    "load_scenario",
    "run_scenario",
]
