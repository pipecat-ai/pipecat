#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Simulated scenario file format for Pipecat behavioral evaluations.

A simulation describes a *caller* rather than a script: who they are, what they
want, and how the outcome is judged. An autonomous persona LLM holds the
conversation with the bot, so the path through it is the bot's and the persona's
to make, not the file's. It is the other kind of scenario file: a manifest lists
simulations under ``scenarios:`` like scripted ones, and ``pipecat eval run``
takes either; a file with a ``persona:`` is a simulation (see
:func:`~pipecat.evals.scenario.load_scenario_file`). Example::

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
        min_quality: 1
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
    the blocks scenarios use (see :mod:`pipecat.evals.script`):
    ``user.modality`` and ``user.speech`` decide whether the persona's turns
    reach the bot as synthesized speech or as text; ``judge.modality``,
    ``judge.transcription`` and ``judge.eval`` decide whether the bot speaks and
    which LLM judges the outcome. Audio modality needs a ``user.speech`` block,
    since every persona turn is synthesized.

``success``
    what counts as the bot having done its job, judged over the whole
    conversation; prose, as long as it needs to be. It is the bot's side of the
    ``goal``: usually that the caller got what they asked for, but where the
    right outcome is to refuse, to qualify, or to escalate, it says so. The run
    succeeded if the judge says yes.

``metrics``
    judged quality criteria, each with ``name``, ``criterion``, and an optional
    ``min_quality`` in 0..1. A criterion says what every reply of the bot should
    be; the judge decides it for each bot turn, in the light of the conversation
    before it and the tool calls the bot had made by then, and the metric's
    score is the share of turns that passed: 0.80 is four replies in five. A
    metric with a ``min_quality`` fails the run when its score is below it; one
    without is reported and never fails anything. The run's ``quality`` is the
    plain mean of the scores. Something the bot must do once, read the order
    back, belongs in ``success``, not here.

    A metric can measure instead of judge: ``measure`` names one of
    ``SIMULATION_MEASURES`` and ``min_value`` / ``max_value`` (at least one)
    bound it; the ``name`` defaults to the measure. The harness computes the
    value from the run, no judge involved, and the metric scores 1 inside the
    range and 0 outside, which fails the run. ``turns`` is the persona's
    turns, ``duration`` the conversation's seconds from its first line to the
    hang-up, ``interruptions`` how often the bot reported being cut off,
    ``words`` the longest bot reply in words, and ``latency`` the slowest
    reply in seconds: from the persona's send to the reply's first token in
    text mode, from the bot noticing the persona stop to its first spoken
    sentence in audio mode. The per-reply measures bound every reply.

``max_turns``, ``max_duration_s``
    backstops on the persona's turns (default 20) and on the run's wall clock
    (default 300 s). A run they end has not succeeded.

``runs``
    how many times the suite runs the simulation (default 1). Every run must
    pass: a persona does not say the same thing twice, so one run is an
    anecdote and three are a check.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipecat.evals.scenario_config import (
    _CFG_EVAL,
    _CFG_LIMIT,
    _DEFAULT_JUDGE,
    _config_lines,
    _ConfigLine,
    _judge_segments,
    _parse_judge_block,
    _parse_user_block,
    _user_segments,
)
from pipecat.evals.scenario_loader import _load_mapping

DEFAULT_MAX_TURNS = 20
DEFAULT_MAX_DURATION_S = 300.0

# What a measured metric can measure, computed by the harness from the run.
SIMULATION_MEASURES = ("turns", "duration", "interruptions", "words", "latency")


@dataclass
class EvalSimulationMetric:
    """A quality metric: a judged criterion, or a measure with a range.

    Parameters:
        name: The metric's name in the results.
        criterion: What the judge decides on each bot turn; ``None`` for a
            measured metric.
        min_quality: The score, in 0..1, below which a judged metric fails the
            run; ``None`` reports the score without gating.
        measure: One of ``SIMULATION_MEASURES``; ``None`` for a judged metric.
        min_value: The measured value's lower bound, inclusive, or ``None``.
        max_value: The measured value's upper bound, inclusive, or ``None``.
    """

    name: str
    criterion: str | None = None
    min_quality: float | None = None
    measure: str | None = None
    min_value: float | None = None
    max_value: float | None = None


@dataclass
class EvalSimulationScenario:
    """A parsed simulation file.

    Parameters:
        name: The simulation name (from ``name:``).
        persona: Who the caller is, as free text for the persona LLM.
        goal: What the caller wants from the call.
        simulator: The persona LLM config (``service``, ``model``, optional
            ``endpoint`` / ``extra``), the same shape as ``judge.eval``.
        success: What counts as the bot having done its job, for the judge.
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
        runs: How many times the suite runs the simulation; every run must pass.
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
            trigger_disconnect=bool(data.get("trigger_disconnect", False)),
            source_path=path,
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
        criterion, measure = item.get("criterion"), item.get("measure")
        if (criterion is None) == (measure is None):
            raise ValueError(
                f"{path}: metric #{idx} is judged ('criterion:') or measured ('measure:'), "
                "one of the two"
            )
        name = item.get("name", measure)
        if not name or not isinstance(name, str):
            raise ValueError(f"{path}: metric #{idx} needs a 'name:'")
        if name in {m.name for m in metrics}:
            raise ValueError(f"{path}: metric {name!r} is listed twice")
        if measure is not None:
            metrics.append(_parse_measure(item, name, measure, path))
            continue
        if not criterion or not isinstance(criterion, str):
            raise ValueError(f"{path}: metric {name!r} needs a 'criterion:' for the judge")
        min_quality = item.get("min_quality")
        if min_quality is not None and (
            isinstance(min_quality, bool)
            or not isinstance(min_quality, (int, float))
            or not 0 <= min_quality <= 1
        ):
            raise ValueError(f"{path}: metric {name!r} 'min_quality:' must be a number in 0..1")
        metrics.append(
            EvalSimulationMetric(
                name=name,
                criterion=criterion,
                min_quality=None if min_quality is None else float(min_quality),
            )
        )
    return metrics


def _parse_measure(item: dict, name: str, measure: Any, path: Path) -> EvalSimulationMetric:
    """Parse a measured metric: a known measure and at least one bound."""
    if measure not in SIMULATION_MEASURES:
        raise ValueError(
            f"{path}: metric {name!r} 'measure:' must be one of {', '.join(SIMULATION_MEASURES)}"
        )
    if "min_quality" in item:
        raise ValueError(
            f"{path}: metric {name!r} is measured; it takes a range, not 'min_quality:'"
        )
    bounds = {}
    for key in ("min_value", "max_value"):
        value = item.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"{path}: metric {name!r} '{key}:' must be a number")
        bounds[key] = None if value is None else float(value)
    if bounds["min_value"] is None and bounds["max_value"] is None:
        raise ValueError(f"{path}: metric {name!r} needs a 'min_value:' or a 'max_value:'")
    return EvalSimulationMetric(name=name, measure=measure, **bounds)


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

    The lines of :func:`~pipecat.evals.scenario_config.describe_config`, the ``user`` line
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
