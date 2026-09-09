#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Simulated scenario file format for Pipecat behavioral evaluations.

A simulation describes a caller rather than a script: who they are, what
they want, and how the outcome is judged. A persona LLM holds the
conversation with the bot on its own. A file with a ``persona:`` is a
simulation, and a manifest lists them under ``scenarios:`` like scripted
ones. Example::

    name: capital_curious
    persona: |
      A curious, polite traveler who asks one thing at a time.
    goal: "Find out what the capital of Germany is, then say goodbye."
    judge: !include judge_text.yaml
    success: "the bot told the caller that the capital of Germany is Berlin"
    metrics:
      - name: politeness
        criterion: "the bot stayed courteous throughout"
        min_score: 1
    max_turns: 10

Fields:

``persona``, ``goal``
    the caller's character and what they are trying to accomplish; both go into
    the persona LLM's instructions (see :mod:`pipecat.evals.persona`).

``simulator``
    the persona LLM: ``service`` (``ollama`` or ``openai``), ``model``, and the
    optional ``endpoint`` / ``extra`` the judge config also takes. Omitted, the
    persona runs on the same local Ollama model as the default judge. The
    model must support function calling: the persona ends the call by calling
    its ``end_call`` tool.

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
    succeeded if the judge says yes. The judge sees the bot's tool calls (name
    and arguments), not their results. Whether the bot made a call at all is a
    ``function_calls`` metric, no judge needed; if a reply must match backend
    data, write the expected value into the criterion ("the reply says the
    appointment is on Tuesday September fifteenth") and keep the mocks
    deterministic so it stays true across runs.

``metrics``
    judged quality criteria, each with ``name``, ``criterion``, and an optional
    ``min_score`` in 0..1. A criterion says what every reply of the bot should
    be; the judge decides it for each bot turn, in the light of the conversation
    before it and the tool calls the bot had made by then, with a yes or a no,
    never a partial score. The metric's score is the share of turns that got a
    yes: 0.80 is four replies in five. A turn the judge leaves out counts as a
    no, recorded as a verdict of ``none`` so a sweep can tell judge trouble
    from bot trouble, and a run with no bot turn has no score and passes. A
    metric with a ``min_score`` fails the run when its score is below it; one
    without is reported and never fails anything. Something the bot must do once, read
    the order back, belongs in ``success``, not here.

    A metric can measure instead of judge: ``measure`` names one of
    ``SIMULATION_MEASURES`` and ``min_value`` / ``max_value`` (at least one)
    bound it; the ``name`` defaults to the measure. The harness computes the
    value from the run, no judge involved, and the metric scores 1 inside the
    range and 0 outside, which fails the run. ``turns`` is the persona's
    turns, ``duration`` the conversation's seconds from its first line to the
    hang-up, ``words`` the longest bot reply in words, and ``latency`` the slowest
    reply in seconds: from the persona's send to the reply's first token in
    text mode, from the bot noticing the persona stop to its first spoken
    sentence in audio mode, which a failure's reason spells out, since the two
    are not comparable. The per-reply measures bound every reply.
    ``function_calls`` takes a ``calls:`` list instead of a range: the calls the
    bot should make, each a name or a ``name`` with ``args`` (a subset of the
    call's arguments), in any order. Every listed call must have happened and
    any call not listed fails it, so ``calls: []`` says the bot must call
    nothing, the check for a caller who should be turned down. A call the bot
    cancelled did not happen.

``max_turns``, ``max_duration_s``, ``max_silence_s``
    backstops on the persona's turns (default 20), on the run's wall clock
    (default 300 s), and on a lull in which neither side does anything
    (default 30 s), so a bot that never greets, or stops answering, ends the
    run as ``silence`` instead of running out the clock. A run they end has
    not succeeded. A failure of the harness's own pipeline, the persona LLM
    first among them, ends the run at once as an error.

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
    _svc_model,
    _user_segments,
)
from pipecat.evals.scenario_loader import _load_mapping
from pipecat.evals.script import EvalFunctionCall

DEFAULT_MAX_TURNS = 20
DEFAULT_MAX_DURATION_S = 300.0
DEFAULT_MAX_SILENCE_S = 30.0

# What a measured metric can measure, computed by the harness from the run.
SIMULATION_MEASURES = ("turns", "duration", "words", "latency", "function_calls")


@dataclass
class EvalSimulationMetric:
    """A quality metric: a judged criterion, or a measure with a range or a call list.

    Parameters:
        name: The metric's name in the results.
        criterion: What the judge decides on each bot turn; ``None`` for a
            measured metric.
        min_score: The share of the bot's turns the judge must answer yes
            for, in 0..1, below which a judged metric fails the run; ``None``
            reports the score without gating.
        measure: One of ``SIMULATION_MEASURES``; ``None`` for a judged metric.
        min_value: The measured value's lower bound, inclusive, or ``None``.
        max_value: The measured value's upper bound, inclusive, or ``None``.
        calls: For ``function_calls``, the calls the bot should make: each a
            name, or a name with ``args`` as a subset of the call's arguments;
            an empty list means none. ``None`` for every other metric.
    """

    name: str
    criterion: str | None = None
    min_score: float | None = None
    measure: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    calls: list[EvalFunctionCall] | None = None


@dataclass
class EvalSimulationScenario:
    """A parsed simulation file.

    Parameters:
        name: The simulation name (from ``name:``).
        persona: Who the caller is, as free text for the persona LLM.
        goal: What the caller wants from the call.
        simulator: The persona LLM config (``service``, ``model``, optional
            ``endpoint`` / ``extra``), the same shape as ``judge.eval``; empty
            for the default local model.
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
        max_silence_s: Cap on a lull with no event from either side, in seconds.
        runs: How many times the suite runs the simulation; every run must pass.
        trigger_disconnect: Whether the harness fires the bot's
            ``on_client_disconnected`` handler when the connection ends.
        source_path: Path the simulation was loaded from, for error messages.
    """

    name: str
    persona: str
    goal: str
    success: str
    simulator: dict = field(default_factory=dict)
    metrics: list[EvalSimulationMetric] = field(default_factory=list)
    judge: dict = field(default_factory=lambda: dict(_DEFAULT_JUDGE))
    bot_audio: bool = False
    transcriber: dict | None = None
    user_audio: bool = False
    user_speech: dict | None = None
    max_turns: int = DEFAULT_MAX_TURNS
    max_duration_s: float = DEFAULT_MAX_DURATION_S
    max_silence_s: float = DEFAULT_MAX_SILENCE_S
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

        simulator = data.get("simulator") or {}
        if not isinstance(simulator, dict):
            raise ValueError(f"{path}: 'simulator:' must be a mapping naming the persona LLM")

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
            max_silence_s=_positive_number(data, "max_silence_s", DEFAULT_MAX_SILENCE_S, path),
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
        min_score = item.get("min_score")
        if min_score is not None and (
            isinstance(min_score, bool)
            or not isinstance(min_score, (int, float))
            or not 0 <= min_score <= 1
        ):
            raise ValueError(f"{path}: metric {name!r} 'min_score:' must be a number in 0..1")
        metrics.append(
            EvalSimulationMetric(
                name=name,
                criterion=criterion,
                min_score=None if min_score is None else float(min_score),
            )
        )
    return metrics


def _parse_measure(item: dict, name: str, measure: Any, path: Path) -> EvalSimulationMetric:
    """Parse a measured metric: a known measure and at least one bound."""
    if measure not in SIMULATION_MEASURES:
        raise ValueError(
            f"{path}: metric {name!r} 'measure:' must be one of {', '.join(SIMULATION_MEASURES)}"
        )
    if "min_score" in item:
        raise ValueError(f"{path}: metric {name!r} is measured; it takes a range, not 'min_score:'")
    if measure == "function_calls":
        return _parse_calls_measure(item, name, path)
    if "calls" in item:
        raise ValueError(
            f"{path}: metric {name!r} takes a range; 'calls:' belongs to 'measure: function_calls'"
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


def _parse_calls_measure(item: dict, name: str, path: Path) -> EvalSimulationMetric:
    """Parse a ``function_calls`` metric: the list of calls the bot should make."""
    if "min_value" in item or "max_value" in item:
        raise ValueError(
            f"{path}: metric {name!r} compares the bot's calls with 'calls:', not a range"
        )
    raw = item.get("calls")
    if not isinstance(raw, list):
        raise ValueError(
            f"{path}: metric {name!r} needs a 'calls:' list, the calls the bot should make "
            "([] for none)"
        )
    calls: list[EvalFunctionCall] = []
    for idx, entry in enumerate(raw):
        if isinstance(entry, str) and entry:
            calls.append(EvalFunctionCall(name=entry))
            continue
        if isinstance(entry, dict) and isinstance(entry.get("name"), str) and entry["name"]:
            args = entry.get("args")
            if args is not None and not isinstance(args, dict):
                raise ValueError(
                    f"{path}: metric {name!r} 'calls:' entry #{idx} 'args:' must be a mapping"
                )
            calls.append(EvalFunctionCall(name=entry["name"], args=args))
            continue
        raise ValueError(
            f"{path}: metric {name!r} 'calls:' entry #{idx} must be a name or a mapping with 'name:'"
        )
    return EvalSimulationMetric(name=name, measure="function_calls", calls=calls)


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

        user  -> modality: text | persona: ollama/gemma4:12b | max_turns: 8 | max_duration_s: 120 | max_silence_s: 30
        judge -> modality: text | eval: ollama/gemma4:12b
        goal  -> Book a table for two at 6 PM, then end the call.

    Args:
        simulation: The parsed simulation to summarize.
        color: When True, ANSI-color the keywords as ``describe_config`` does.

    Returns:
        The summary, one line per section.
    """
    persona = _svc_model({**_DEFAULT_JUDGE, **simulation.simulator}, "ollama", "model")
    user = _user_segments(simulation) + [
        ("persona", persona, _CFG_EVAL),
        ("max_turns", str(simulation.max_turns), _CFG_LIMIT),
        ("max_duration_s", f"{simulation.max_duration_s:g}", _CFG_LIMIT),
        ("max_silence_s", f"{simulation.max_silence_s:g}", _CFG_LIMIT),
    ]
    goal = " ".join(simulation.goal.split())
    lines: list[_ConfigLine] = [
        ("user", user),
        ("judge", _judge_segments(simulation)),
        ("goal", goal),
    ]
    return _config_lines(lines, color=color)
