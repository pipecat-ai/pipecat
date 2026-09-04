#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Scenario files for Pipecat behavioral evaluations, of either kind.

A scenario file describes one conversation to hold with a bot and how to judge
it. There are two kinds, and a file's top-level keys say which:

``turns:``
    a *scripted* scenario: the user's turns are written out, each with the
    events expected back from the bot (:mod:`pipecat.evals.script`).

``persona:``
    a *simulated* scenario, a simulation for short: an LLM plays a caller with a
    goal, and a judge reads the whole conversation
    (:mod:`pipecat.evals.simulation`).

Both carry the same ``user:`` and ``judge:`` blocks
(:mod:`pipecat.evals.scenario_config`) and are read by the same YAML loader with
``!include`` support (:mod:`pipecat.evals.scenario_loader`). This module gathers
the public names of both kinds, and :func:`load_scenario_file` loads a file as
whichever kind it is.
"""

from pathlib import Path

from pipecat.evals.scenario_config import EvalConfigured, describe_config
from pipecat.evals.scenario_loader import _load_mapping
from pipecat.evals.script import (
    FUNCTION_CALL_EVENTS,
    JUDGEABLE_EVENTS,
    EvalExpectation,
    EvalFunctionCall,
    EvalScenario,
    EvalScriptScenario,
    EvalScriptTurn,
    EvalSendAfter,
    EvalTurn,
)
from pipecat.evals.simulation import (
    EvalSimulationMetric,
    EvalSimulationScenario,
    describe_simulation,
)

__all__ = [
    "FUNCTION_CALL_EVENTS",
    "JUDGEABLE_EVENTS",
    "EvalConfigured",
    "EvalExpectation",
    "EvalFunctionCall",
    "EvalScenario",
    "EvalScriptScenario",
    "EvalScriptTurn",
    "EvalSendAfter",
    "EvalSimulationMetric",
    "EvalSimulationScenario",
    "EvalTurn",
    "describe_config",
    "describe_simulation",
    "load_scenario_file",
]


def load_scenario_file(path: str | Path) -> EvalScriptScenario | EvalSimulationScenario:
    """Load a scenario file as whichever kind it is.

    A file with a ``persona:`` is a simulation; one with ``turns:`` is a scripted
    scenario. This is what a manifest's ``scenarios:`` entries and ``pipecat eval
    run`` load through, so the two kinds mix freely in one list.

    Args:
        path: Path to a scenario or simulation YAML file.

    Returns:
        The parsed :class:`~pipecat.evals.simulation.EvalSimulationScenario` or
        :class:`~pipecat.evals.script.EvalScriptScenario`.

    Raises:
        ValueError: If the file is neither kind, claims to be both, or is
            invalid for its kind.
        FileNotFoundError: If the path doesn't exist.
    """
    path = Path(path)
    data = _load_mapping(path)
    if "persona" in data and "turns" in data:
        raise ValueError(
            f"{path}: a scenario is scripted ('turns:') or a simulation ('persona:'), not both"
        )
    if "persona" in data:
        return EvalSimulationScenario.load(path)
    if "turns" in data:
        return EvalScriptScenario.load(path)
    raise ValueError(
        f"{path}: a scenario file needs 'turns:' (scripted) or 'persona:' (a simulation)"
    )
