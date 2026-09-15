#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Scenario files for Pipecat behavioral evaluations, of either kind.

A scenario describes one conversation to hold with a bot and how to judge it.
A file holds one or more of them under ``scenarios:``, each with a ``name:``,
and a scenario's keys say which kind it is:

``turns:``
    a *scripted* scenario: the user's turns are written out, each with the
    events expected back from the bot (:mod:`pipecat.evals.script`).

``persona:``
    a *simulated* scenario, a simulation for short: an LLM plays a caller with a
    goal, and a judge reads the whole conversation
    (:mod:`pipecat.evals.simulation`).

Both carry the same ``user:`` and ``judge:`` blocks
(:mod:`pipecat.evals.scenario_config`) and are read by the same YAML loader with
``!include`` support (:mod:`pipecat.evals.scenario_loader`).

The file's other top-level keys are defaults for its scenarios, and a scenario
that sets the same key replaces the value as a whole (a ``context:`` is restated
in full, never appended to). A scenario is named ``<file name>/<scenario name>``.
Most files hold one scenario; a file holds several to test one behavior
through many short conversations::

    name: turn_completion
    judge: !include ../judge_text.yaml
    context:
      - role: system
        content: "You are a travel assistant."

    scenarios:
      - name: short_answer
        turns:
          - user: "Japan."
            expect:
              - event: response
      - name: with_history
        context:                      # replaces the file's context
          - role: system
            content: "You are a travel assistant."
          - role: assistant
            content: "Where would you go?"
        turns:
          - user: "Japan."
            expect:
              - event: response

The scenarios of a file are independent: each runs on its own, against its
own bot.

.. deprecated:: 1.11.0
    Use a ``scenarios:`` list instead of a scenario's own keys (``turns:`` or
    ``persona:``) at a file's top level. Such a file still loads, as that one
    scenario under the file's ``name:``, with a ``DeprecationWarning``. Will be
    removed in 2.0.0.

This module gathers the public names of both kinds, :func:`load_scenarios`
loads a file as the scenarios it holds, :func:`load_scenario` picks one of
them, and :func:`is_scenario_file` tells a scenario from a fragment it
includes.
"""

import warnings
from enum import StrEnum
from pathlib import Path

import yaml

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
from pipecat.utils.deprecation import deprecated

__all__ = [
    "FUNCTION_CALL_EVENTS",
    "JUDGEABLE_EVENTS",
    "EvalConfigured",
    "EvalKind",
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
    "is_scenario_file",
    "load_scenario",
    "load_scenario_file",
    "load_scenarios",
]


class EvalKind(StrEnum):
    """The two kinds of scenario, as a run, a session, and a results record name them."""

    SCRIPT = "script"
    SIMULATION = "simulation"


EvalLoadedScenario = EvalScriptScenario | EvalSimulationScenario


def load_scenarios(path: str | Path) -> list[EvalLoadedScenario]:
    """Load the scenarios a file holds, each as whichever kind it is.

    Manifests and ``pipecat eval run`` load through here, so the two kinds mix
    in one list. A file in the deprecated shape, a scenario's own keys at the
    top level and no ``scenarios:``, loads as that one scenario under the
    file's ``name:`` and warns.

    Args:
        path: Path to a scenario YAML file.

    Returns:
        The parsed scenarios, in file order.

    Raises:
        ValueError: If the file is malformed, or a scenario is neither kind,
            claims to be both, or is invalid for its kind.
        FileNotFoundError: If the path doesn't exist.
    """
    path = Path(path)
    data = _load_mapping(path)
    entries = data.get("scenarios")
    if entries is None:
        warnings.warn(
            f"{path}: a scenario file's top level holding 'turns:' or 'persona:' is deprecated "
            "since 1.11.0 and will be removed in 2.0.0. Put the scenario under a 'scenarios:' "
            "list instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return [_scenario_from_mapping(data, path)]

    group = data.get("name")
    if not group or not isinstance(group, str):
        raise ValueError(f"{path}: missing or invalid 'name:' field")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: 'scenarios:' must be a non-empty list")
    defaults = {key: value for key, value in data.items() if key != "scenarios"}

    scenarios: list[EvalLoadedScenario] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: scenario #{idx} must be a mapping")
        if "scenarios" in entry:
            raise ValueError(f"{path}: scenario #{idx} cannot hold a 'scenarios:' list of its own")
        name = entry.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(f"{path}: scenario #{idx} needs a 'name:'")
        merged = {**defaults, **entry, "name": f"{group}/{name}"}
        scenarios.append(_scenario_from_mapping(merged, path))

    names = [scenario.name for scenario in scenarios]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate scenario names: {', '.join(duplicates)}")
    return scenarios


def load_scenario(path: str | Path, name: str | None = None) -> EvalLoadedScenario:
    """Load one scenario of a file: the one called ``name``, or the only one.

    Args:
        path: Path to a scenario YAML file.
        name: The scenario to pick, as :func:`load_scenarios` names it
            (``<file name>/<scenario name>``); ``None`` for a file holding one.

    Returns:
        The parsed :class:`~pipecat.evals.simulation.EvalSimulationScenario` or
        :class:`~pipecat.evals.script.EvalScriptScenario`.

    Raises:
        ValueError: If the file is invalid, holds several scenarios and no
            ``name`` picks one, or holds none called ``name``.
        FileNotFoundError: If the path doesn't exist.
    """
    scenarios = load_scenarios(path)
    if name is None:
        if len(scenarios) == 1:
            return scenarios[0]
        raise ValueError(f"{path}: holds {len(scenarios)} scenarios; pick one by name")
    for scenario in scenarios:
        if scenario.name == name:
            return scenario
    names = ", ".join(scenario.name for scenario in scenarios)
    raise ValueError(f"{path}: no scenario called {name!r} (has {names})")


@deprecated(
    "`load_scenario_file` is deprecated since 1.11.0 and will be removed in 2.0.0. "
    "Use `load_scenarios` instead."
)
def load_scenario_file(path: str | Path) -> EvalLoadedScenario:
    """Load a file holding one scenario, as whichever kind it is.

    .. deprecated:: 1.11.0
        Use :func:`load_scenarios` instead, which returns every scenario a file
        holds; :func:`load_scenario` picks one. Will be removed in 2.0.0.

    Args:
        path: Path to a scenario or simulation YAML file.

    Returns:
        The parsed scenario.
    """
    return load_scenario(path)


def _scenario_from_mapping(data: dict, path: Path) -> EvalLoadedScenario:
    """Parse one scenario's mapping as a simulation (``persona:``) or a script (``turns:``)."""
    if "persona" in data and "turns" in data:
        raise ValueError(
            f"{path}: a scenario is scripted ('turns:') or a simulation ('persona:'), not both"
        )
    if "persona" in data:
        return EvalSimulationScenario.from_mapping(data, path)
    if "turns" in data:
        return EvalScriptScenario.from_mapping(data, path)
    raise ValueError(f"{path}: a scenario needs 'turns:' (scripted) or 'persona:' (a simulation)")


def is_scenario_file(path: str | Path) -> bool:
    """Whether a YAML file is a scenario of either kind, rather than a fragment one includes.

    Every scenario has a ``name:``; an included fragment has none. A file that
    does not parse counts as a scenario, so loading it reports the error
    instead of a directory run skipping it silently.

    Args:
        path: Path to a YAML file.

    Returns:
        True unless the file parses to a mapping without a ``name``.
    """
    try:
        return "name" in _load_mapping(Path(path))
    except (ValueError, OSError, yaml.YAMLError):
        return True
