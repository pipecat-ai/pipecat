"""Behavioral harness for Pipecat bots.

This package provides a scenario-based test runner that drives a bot via the
eval transport and asserts on the semantic event stream it emits. See
:mod:`pipecat.evals.scenario` for the YAML schema and
:mod:`pipecat.evals.harness` for the runner.
"""

from pipecat.evals.harness import (
    EvalAssertionFailure,
    EvalResult,
    EvalSession,
    EvalTurnProgress,
    run_scenario,
)
from pipecat.evals.scenario import (
    EvalExpectation,
    EvalFunctionCall,
    EvalScenario,
    EvalSendAfter,
    EvalTurn,
    load_scenario,
)
from pipecat.evals.transcribe import EvalTranscriber
from pipecat.evals.voice import EvalVoice

__all__ = [
    "EvalAssertionFailure",
    "EvalResult",
    "EvalSession",
    "EvalTranscriber",
    "EvalVoice",
    "EvalTurnProgress",
    "EvalExpectation",
    "EvalFunctionCall",
    "EvalScenario",
    "EvalSendAfter",
    "EvalTurn",
    "load_scenario",
    "run_scenario",
]
