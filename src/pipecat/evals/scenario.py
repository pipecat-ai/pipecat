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
(:mod:`pipecat.evals.scenario_config`). A file is read by a ``SafeLoader`` that
resolves only plain-decimal integers, so a DTMF ``012`` keeps its digits, with
an ``!include <path>`` tag that splices in another YAML file relative to the
including one, so files can share their ``user:`` and ``judge:`` blocks.

Any key a scenario can have may also sit at the top of the file. There it is
the default for every scenario in the file. A scenario that sets the same key
replaces the whole value; nothing is merged, so a ``context:`` is written out in
full, never added to. A scenario is named ``<file name>/<scenario name>``.

Most files hold one scenario. A file holds several when they test one behavior
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

``turns:`` and ``persona:`` may sit at the top too. That is for a file whose
scenarios hold the same conversation and differ in one thing only. Two common
shapes:

The same turns, judged differently. The turns are written once, and each
scenario names its own judge or modality, so every judge sees exactly the same
conversation::

    name: interruption
    turns:
      - user: "Tell me a long story about Paris."
        expect:
          - event: llm_started
      - user: "Actually, what's the capital of Japan?"
        send_after: {event: llm_started, delay_ms: 2000}
        expect:
          - event: bot_interrupted
          - event: response
            eval: "says Tokyo instead of continuing the story"

    scenarios:
      - name: text
        judge: !include ../judge_text.yaml
      - name: audio
        user: !include ../user_audio.yaml
        judge: !include ../judge_audio.yaml

The same caller, with different goals. The persona and what counts as success
are written once, and each scenario gives the caller a different errand::

    name: diner
    persona: "Jamie, calling a restaurant. Friendly and to the point."
    success: "the bot did what the caller asked and confirmed it"

    scenarios:
      - name: book
        goal: "Book a table for two at 6 PM tonight, then end the call."
      - name: cancel
        goal: "Cancel tonight's booking under the name Jamie, then end the call."

The scenarios of a file are independent: each runs on its own, against its
own bot.

.. deprecated:: 1.11.0
    Use a ``scenarios:`` list instead of a scenario's own keys (``turns:`` or
    ``persona:``) at a file's top level. Such a file still loads, as that one
    scenario under the file's ``name:``, with a ``DeprecationWarning``. Will be
    removed in 2.0.0.

This module gathers the public names of both kinds, :class:`EvalScenarioFile`
loads a file as the scenarios it holds, and :func:`is_scenario_file` tells a
scenario from a fragment it includes.
"""

import re
import warnings
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from pipecat.evals.scenario_config import EvalConfigured, describe_config
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
    _parse_script,
)
from pipecat.evals.simulation import (
    EvalSimulationMetric,
    EvalSimulationScenario,
    _parse_simulation,
    describe_simulation,
)
from pipecat.utils.deprecation import deprecated
from pipecat.utils.yaml import include_loader

__all__ = [
    "FUNCTION_CALL_EVENTS",
    "JUDGEABLE_EVENTS",
    "EvalConfigured",
    "EvalKind",
    "EvalExpectation",
    "EvalFunctionCall",
    "EvalScenario",
    "EvalScenarioFile",
    "EvalScriptScenario",
    "EvalScriptTurn",
    "EvalSendAfter",
    "EvalSimulationMetric",
    "EvalSimulationScenario",
    "EvalTurn",
    "describe_config",
    "describe_simulation",
    "is_scenario_file",
    "load_scenario_file",
]


class _ScenarioLoader(yaml.SafeLoader):
    """A SafeLoader that reads only plain decimal numbers as ints.

    YAML 1.1 would read ``010`` as octal and ``0x10`` as hex, which rewrites a
    DTMF sequence before the scenario sees it. With those resolvers dropped,
    ``dtmf: 123`` still loads as an int and ``dtmf: 012`` stays a string.
    """


# Strip the inherited int resolvers (which match octal/hex/binary/sexagesimal)
# and register a decimal-only replacement. Underscores stay allowed to match
# YAML's grouping syntax (e.g. ``1_000``); a leading zero (``012``) no longer
# matches, so such tokens load as strings.
_ScenarioLoader.yaml_implicit_resolvers = {
    ch: [(tag, rx) for tag, rx in resolvers if tag != "tag:yaml.org,2002:int"]
    for ch, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
yaml.add_implicit_resolver(
    "tag:yaml.org,2002:int",
    re.compile(r"^[-+]?(?:0|[1-9][0-9_]*)$"),
    list("-+0123456789"),
    Loader=_ScenarioLoader,
)


class EvalKind(StrEnum):
    """The two kinds of scenario, as a run, a session, and a results record name them."""

    SCRIPT = "script"
    SIMULATION = "simulation"


EvalLoadedScenario = EvalScriptScenario | EvalSimulationScenario


@dataclass
class EvalScenarioFile:
    """A scenario file: what it is called, where it is, and the scenarios it holds.

    Manifests and ``pipecat eval run`` load files through :meth:`load`, so the
    two kinds of scenario mix in one list. ``file[name]`` picks a scenario by
    its ``<file name>/<scenario name>``.

    Parameters:
        name: The file's ``name:``.
        path: The file it was read from.
        scenarios: The scenarios it holds, in file order.
    """

    name: str
    path: Path
    scenarios: list[EvalLoadedScenario]

    @classmethod
    def load(cls, path: str | Path) -> "EvalScenarioFile":
        """Read a scenario file, parsing each scenario as whichever kind it is.

        A file in the deprecated shape, a scenario's own keys at the top level
        and no ``scenarios:``, loads as that one scenario under the file's
        ``name:`` and warns.

        Args:
            path: Path to a scenario YAML file.

        Returns:
            The loaded file.

        Raises:
            ValueError: If the file is malformed, or a scenario is neither kind,
                claims to be both, or is invalid for its kind.
            FileNotFoundError: If the path doesn't exist.
        """
        path = Path(path)
        data = _load_mapping(path)
        name = data.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(f"{path}: missing or invalid 'name:' field")

        entries = data.get("scenarios")
        if entries is None:
            warnings.warn(
                f"{path}: a scenario file's top level holding 'turns:' or 'persona:' is "
                "deprecated since 1.11.0 and will be removed in 2.0.0. Put the scenario under a "
                "'scenarios:' list instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return cls(name=name, path=path, scenarios=[_scenario_from_mapping(data, path)])

        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{path}: 'scenarios:' must be a non-empty list")
        defaults = {key: value for key, value in data.items() if key != "scenarios"}

        scenarios: list[EvalLoadedScenario] = []
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: scenario #{idx} must be a mapping")
            if "scenarios" in entry:
                raise ValueError(
                    f"{path}: scenario #{idx} cannot hold a 'scenarios:' list of its own"
                )
            entry_name = entry.get("name")
            if not entry_name or not isinstance(entry_name, str):
                raise ValueError(f"{path}: scenario #{idx} needs a 'name:'")
            merged = {**defaults, **entry, "name": f"{name}/{entry_name}"}
            scenarios.append(_scenario_from_mapping(merged, path))

        names = [scenario.name for scenario in scenarios]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"{path}: duplicate scenario names: {', '.join(duplicates)}")
        return cls(name=name, path=path, scenarios=scenarios)

    def __getitem__(self, name: str) -> EvalLoadedScenario:
        """The scenario called ``name``.

        Raises:
            KeyError: If the file holds no scenario of that name.
        """
        for scenario in self.scenarios:
            if scenario.name == name:
                return scenario
        names = ", ".join(scenario.name for scenario in self.scenarios)
        raise KeyError(f"{self.path}: no scenario called {name!r} (has {names})")

    def __iter__(self):
        """Iterate over the scenarios, in file order."""
        return iter(self.scenarios)

    def __len__(self) -> int:
        """How many scenarios the file holds."""
        return len(self.scenarios)


@deprecated(
    "`load_scenario_file` is deprecated since 1.11.0 and will be removed in 2.0.0. "
    "Use `EvalScenarioFile.load` instead."
)
def load_scenario_file(path: str | Path) -> EvalLoadedScenario:
    """Load a file holding one scenario, as whichever kind it is.

    .. deprecated:: 1.11.0
        Use :meth:`EvalScenarioFile.load` instead, which returns every scenario
        a file holds. Will be removed in 2.0.0.

    Args:
        path: Path to a scenario or simulation YAML file.

    Returns:
        The parsed scenario.

    Raises:
        ValueError: If the file holds several scenarios, or is invalid.
    """
    scenarios = EvalScenarioFile.load(path).scenarios
    if len(scenarios) != 1:
        raise ValueError(f"{path}: holds {len(scenarios)} scenarios; use EvalScenarioFile.load()")
    return scenarios[0]


def _load_mapping(path: Path) -> dict:
    """Load a scenario file's top-level mapping, resolving ``!include`` tags relative to the file.

    Raises:
        ValueError: If the top level is not a mapping.
    """
    with path.open() as f:
        data = yaml.load(f, include_loader(path.parent, base=_ScenarioLoader))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def _scenario_from_mapping(data: dict, path: Path) -> EvalLoadedScenario:
    """Parse one scenario's mapping as a simulation (``persona:``) or a script (``turns:``)."""
    if "persona" in data and "turns" in data:
        raise ValueError(
            f"{path}: a scenario is scripted ('turns:') or a simulation ('persona:'), not both"
        )
    if "persona" in data:
        return _parse_simulation(data, path)
    if "turns" in data:
        return _parse_script(data, path)
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
