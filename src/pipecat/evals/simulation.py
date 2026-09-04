#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Simulation file format for persona-driven Pipecat evaluations.

A simulation describes a *caller* rather than a script: who they are, what they
want, and how the outcome is judged. An autonomous persona LLM holds the
conversation with the bot, so the path through it is the bot's and the persona's
to make, not the file's. It is the other kind of scenario file: a manifest lists
simulations under ``scenarios:`` like scripted ones, and ``pipecat eval run``
takes either; a file with a ``persona:`` is a simulation (see
:func:`load_scenario_file`). Example::

    name: capital_curious
    persona: |
      A curious, polite traveler who asks one thing at a time.
    goal: "Find out what the capital of Germany is, then say goodbye."
    simulator:
      service: openai
      model: gpt-4o-mini
    judge: !include judge_text.yaml
    success: "the bot told the caller that the capital of Germany is Berlin"
    metrics:
      - name: politeness
        criterion: "the bot stayed courteous throughout"
    max_turns: 10

Fields:

``persona``, ``goal``
    the caller's character and what they are trying to accomplish; both go into
    the persona LLM's instructions (see :mod:`pipecat.evals.persona`).

``simulator``
    the persona LLM: ``service`` (``openai`` or ``ollama``), ``model``, and the
    optional ``endpoint`` / ``extra`` the judge config also takes. The model
    must support function calling: the persona ends the call by calling its
    ``end_call`` tool.

``user``, ``judge``
    the blocks scenarios use (see :mod:`pipecat.evals.scenario`):
    ``user.modality`` and ``user.speech`` decide whether the persona's turns
    reach the bot as synthesized speech or as text; ``judge.modality``,
    ``judge.transcription`` and ``judge.eval`` decide whether the bot speaks and
    which LLM judges the outcome. Audio modality needs a ``user.speech`` block,
    since every persona turn is synthesized.

``success``
    the goal criterion the judge decides over the whole conversation; the run
    succeeded if the judge says yes.

``metrics``
    judged quality criteria, each with ``name``, ``criterion`` and an optional
    ``weight`` (default 1). Each scores 1 (the judge says yes) or 0, and they
    roll up into the run's ``quality`` as a weighted mean.

``max_turns``, ``max_duration_s``
    backstops on the persona's turns (default 20) and on the run's wall clock
    (default 300 s). A run they end has not succeeded.

``runs``, ``pass_threshold``
    how many times the suite runs the simulation (default 1) and the success
    rate it needs to pass (default 1.0).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipecat.evals.scenario import (
    _CFG_EVAL,
    _CFG_LIMIT,
    _DEFAULT_JUDGE,
    EvalScriptScenario,
    _config_lines,
    _ConfigLine,
    _judge_segments,
    _load_mapping,
    _parse_judge_block,
    _parse_user_block,
    _user_segments,
)

DEFAULT_MAX_TURNS = 20
DEFAULT_MAX_DURATION_S = 300.0


@dataclass
class EvalSimulationMetric:
    """A judged quality criterion.

    Parameters:
        name: The metric's name in the results.
        criterion: What the judge decides over the whole conversation.
        weight: Its weight in the run's ``quality``.
    """

    name: str
    criterion: str
    weight: float = 1.0


@dataclass
class EvalSimulationScenario:
    """A parsed simulation file.

    Parameters:
        name: The simulation name (from ``name:``).
        persona: Who the caller is, as free text for the persona LLM.
        goal: What the caller wants from the call.
        simulator: The persona LLM config (``service``, ``model``, optional
            ``endpoint`` / ``extra``), the same shape as ``judge.eval``.
        success: The goal criterion the judge decides over the conversation.
        metrics: The judged quality criteria.
        judge: Judge LLM config, as for a scenario.
        bot_audio: Whether the bot speaks (``judge.modality: audio``).
        transcriber: STT config for the bot's audio in audio modality, else None.
        user_audio: Whether the persona's turns reach the bot as speech
            (``user.modality: audio``).
        user_speech: TTS config the persona's turns are synthesized with in
            audio modality, else None.
        max_turns: Cap on the persona's turns.
        max_duration_s: Cap on the run's wall clock, in seconds.
        runs: How many times the suite runs the simulation.
        pass_threshold: Success rate the suite needs to pass the simulation.
        trigger_disconnect: Whether the harness fires the bot's
            ``on_client_disconnected`` handler when the connection ends.
        source_path: Path the simulation was loaded from, for error messages.
    """

    name: str
    persona: str
    goal: str
    simulator: dict
    success: str
    metrics: list[EvalSimulationMetric] = field(default_factory=list)
    judge: dict = field(default_factory=lambda: dict(_DEFAULT_JUDGE))
    bot_audio: bool = False
    transcriber: dict | None = None
    user_audio: bool = False
    user_speech: dict | None = None
    max_turns: int = DEFAULT_MAX_TURNS
    max_duration_s: float = DEFAULT_MAX_DURATION_S
    runs: int = 1
    pass_threshold: float = 1.0
    trigger_disconnect: bool = False
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "EvalSimulationScenario":
        """Parse a simulation YAML file into an :class:`EvalSimulationScenario`.

        Args:
            path: Path to a YAML file with the simulation schema.

        Returns:
            The parsed simulation.

        Raises:
            ValueError: If the file structure is invalid.
            FileNotFoundError: If the path doesn't exist.
        """
        path = Path(path)
        data = _load_mapping(path)

        def text(key: str) -> str:
            value = data.get(key)
            if not value or not isinstance(value, str) or not value.strip():
                raise ValueError(f"{path}: missing or invalid '{key}:' field (a non-empty string)")
            return value.strip()

        simulator = data.get("simulator")
        if not isinstance(simulator, dict) or not simulator.get("service"):
            raise ValueError(
                f"{path}: 'simulator:' must be a mapping naming the persona LLM "
                "(at least 'service:')"
            )

        user_audio, user_speech = _parse_user_block(data.get("user"), path)
        if user_audio and user_speech is None:
            raise ValueError(
                f"{path}: 'user.modality: audio' requires a 'user.speech:' block "
                "(TTS service + voice) to synthesize the persona's turns"
            )
        bot_audio, transcriber, judge = _parse_judge_block(data.get("judge"), path)

        return cls(
            name=text("name"),
            persona=text("persona"),
            goal=text("goal"),
            simulator=simulator,
            success=text("success"),
            metrics=_parse_metrics(data.get("metrics"), path),
            judge=judge,
            bot_audio=bot_audio,
            transcriber=transcriber,
            user_audio=user_audio,
            user_speech=user_speech,
            max_turns=_positive_int(data, "max_turns", DEFAULT_MAX_TURNS, path),
            max_duration_s=_positive_number(data, "max_duration_s", DEFAULT_MAX_DURATION_S, path),
            runs=_positive_int(data, "runs", 1, path),
            pass_threshold=_positive_number(data, "pass_threshold", 1.0, path),
            trigger_disconnect=bool(data.get("trigger_disconnect", False)),
            source_path=path,
        )


def load_scenario_file(path: str | Path) -> EvalScriptScenario | EvalSimulationScenario:
    """Load a scenario file as whichever kind it is.

    A file with a ``persona:`` is a simulation; one with ``turns:`` is a scripted
    scenario. This is what a manifest's ``scenarios:`` entries and ``pipecat eval
    run`` load through, so the two kinds mix freely in one list.

    Args:
        path: Path to a scenario or simulation YAML file.

    Returns:
        The parsed :class:`EvalSimulationScenario` or
        :class:`~pipecat.evals.scenario.EvalScriptScenario`.

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


def _parse_metrics(raw: Any, path: Path) -> list[EvalSimulationMetric]:
    """Parse the ``metrics:`` list."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: 'metrics:' must be a list")
    metrics: list[EvalSimulationMetric] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: metric #{idx} must be a mapping")
        name = item.get("name")
        criterion = item.get("criterion")
        if not name or not isinstance(name, str):
            raise ValueError(f"{path}: metric #{idx} needs a 'name:'")
        if not criterion or not isinstance(criterion, str):
            raise ValueError(f"{path}: metric {name!r} needs a 'criterion:' for the judge")
        weight = item.get("weight", 1.0)
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0:
            raise ValueError(f"{path}: metric {name!r} 'weight:' must be a positive number")
        metrics.append(EvalSimulationMetric(name=name, criterion=criterion, weight=float(weight)))
    return metrics


def _positive_int(data: dict, key: str, default: int, path: Path) -> int:
    value = data.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{path}: '{key}:' must be a positive integer")
    return value


def _positive_number(data: dict, key: str, default: float, path: Path) -> float:
    value = data.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path}: '{key}:' must be a positive number")
    return float(value)


def describe_simulation(simulation: EvalSimulationScenario, *, color: bool = False) -> str:
    """Three-line summary of a simulation's user config, judge config, and goal, for pre-run logs.

    The lines of :func:`~pipecat.evals.scenario.describe_config`, the ``user`` line
    also naming the persona LLM that plays the user and the run's caps, then the
    caller's goal, e.g.::

        user  -> modality: text | persona: openai/gpt-4o-mini | max_turns: 8 | max_duration_s: 120
        judge -> modality: text | eval: ollama/gemma4:12b
        goal  -> Book a table for two at 6 PM, then end the call.

    Args:
        simulation: The parsed simulation to summarize.
        color: When True, ANSI-color the keywords as ``describe_config`` does.

    Returns:
        The summary, one line per section.
    """
    persona = f"{simulation.simulator.get('service', '?')}/{simulation.simulator.get('model', '?')}"
    user = _user_segments(simulation) + [
        ("persona", persona, _CFG_EVAL),
        ("max_turns", str(simulation.max_turns), _CFG_LIMIT),
        ("max_duration_s", f"{simulation.max_duration_s:g}", _CFG_LIMIT),
    ]
    goal = " ".join(simulation.goal.split())
    lines: list[_ConfigLine] = [
        ("user", user),
        ("judge", _judge_segments(simulation)),
        ("goal", goal),
    ]
    return _config_lines(lines, color=color)
