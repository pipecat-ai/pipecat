#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Scenario file format for Pipecat behavioral evaluations.

A scenario is a YAML file describing a scripted conversation and the semantic
events expected to flow back from the bot. Simple example::

    name: simple_user_input
    turns:
      - user: "hello world"
        expect:
          - event: user_started_speaking
          - event: user_stopped_speaking
            transcript_contains: "hello world"

The runner (see :mod:`pipecat.evals.harness`) loads the scenario, connects
to the bot's eval transport WebSocket, drives each turn, collects events, and
asserts on them in order.

Supported expectation fields (per event):
    event: <name>              required — event type name
    within_ms: <int>           latency budget from the most recent anchor
    transcript_contains: <str> substring check on user_stopped_speaking.transcript
    text_contains: <str>       substring check on llm_response.text or bot_stopped_speaking.text
    name: <str>                tool_call.name equality
    args: <object>             tool_call.args equality
    eval: <str>                natural-language criterion the event's text content
                               must satisfy, evaluated by a judge LLM (see
                               :mod:`pipecat.evals.judge`). Only meaningful on
                               events with bot-generated text content —
                               ``llm_response`` and ``bot_stopped_speaking``.

A turn may also include ``send_after:`` to schedule its ``user_input`` send
relative to a prior event (used for interruption / barge-in tests).

Top-level optional fields:
    reset:    list of LLM messages the harness sends as a reset before driving
              turns (default empty list, which clears the bot's context).
    judge:    judge LLM configuration block, e.g.::

                  judge:
                    service: ollama
                    model: qwen2.5:3b
                    endpoint: http://localhost:11434/v1   # optional

    fast:     when true, the harness asks the eval transport to skip real-time
              audio pacing. Use for text-only evals that don't care about
              interruption timing — runs as fast as the bot can produce
              tokens. Sent to the bot as ``{"type": "settings", "fast": true}``.
    fixtures: free-form mapping (e.g. ``bot_url:``).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import yaml
except ModuleNotFoundError as e:
    logger.error("PyYAML is required for the scenario runner. Install with: pip install pyyaml")
    raise ImportError(f"Missing module: {e}") from e

# Events whose payloads carry bot-generated text the judge can sensibly
# evaluate. Asserting ``judge:`` on anything else (user transcripts, tool
# calls, interruption signals) produces a parser warning — the test controls
# user input deterministically, so judging it adds cost without signal.
JUDGEABLE_EVENTS = frozenset({"llm_response", "bot_stopped_speaking"})


@dataclass
class Expectation:
    """A single expected event in a scenario turn.

    Parameters:
        event: Required — the semantic event name (e.g. ``user_stopped_speaking``).
        within_ms: Optional latency budget, measured from the most recent anchor
            (typically the preceding ``user_input`` send for the first event of
            a turn, or the previous matched event otherwise).
        transcript_contains: Optional substring check on
            ``user_stopped_speaking.transcript``.
        text_contains: Optional substring check on ``llm_response.text`` or
            ``bot_stopped_speaking.text``.
        name: Optional equality check for ``tool_call.name``.
        args: Optional equality check for ``tool_call.args``.
        eval: Optional natural-language criterion the event's text content
            must satisfy. Evaluated by a judge LLM. Only meaningful on
            ``llm_response`` and ``bot_stopped_speaking``.
        raw: The original parsed dict, for forward compatibility.
    """

    event: str
    within_ms: int | None = None
    transcript_contains: str | None = None
    text_contains: str | None = None
    name: str | None = None
    args: dict | None = None
    eval: str | None = None
    raw: dict = field(default_factory=dict)


@dataclass
class SendAfter:
    """Event-driven scheduling for a turn's ``user_input`` send.

    When set on a :class:`Turn`, the harness waits for ``event`` to have been
    seen (either earlier in the run or arriving now), then waits an additional
    ``delay_ms`` before sending the turn's ``user`` text. Used for barge-in
    tests: ``send_after: {event: bot_started_speaking, delay_ms: 500}`` means
    "interrupt 500ms into the bot's response."

    Parameters:
        event: Name of the event to schedule from.
        delay_ms: Additional delay in milliseconds after the event was received.
    """

    event: str
    delay_ms: int


@dataclass
class Turn:
    """One turn in a scenario.

    A turn is either driven by the harness sending a ``user`` utterance, or it
    is observation-only (no ``user`` field — useful for bot-first scenarios
    like opening greetings).

    Parameters:
        user: Optional text to send as ``{"type": "user_input", "text": ...}``.
            If absent, the turn just waits for and asserts on expected events.
        expect: Expected events, in the order they should arrive.
        send_after: Optional event-driven schedule for when the ``user`` send
            should fire. Only meaningful when ``user`` is set.
    """

    user: str | None
    expect: list[Expectation]
    send_after: SendAfter | None = None


@dataclass
class Scenario:
    """A parsed scenario file.

    Parameters:
        name: The eval name (from ``name:``).
        turns: Ordered list of turns.
        reset: Messages to seed the bot's LLM context with before this eval
            runs. Sent by the harness as a ``{"type": "reset", "messages":
            ...}`` message right after the initial ``ready`` handshake. Empty
            list (the default) clears the context entirely; bots without an
            LLM context aggregator ignore the resulting frame.
        judge: Judge LLM configuration dict with keys ``service``, ``model``,
            and optional ``endpoint``. Defaults to
            ``{"service": "ollama", "model": "qwen2.5:3b"}``.
        fast: When True, the harness asks the eval transport to skip real-time
            audio pacing. Useful for text-only evals.
        fixtures: Optional fixtures dict (e.g. ``bot_url:``).
        source_path: Path the scenario was loaded from, for error messages.
    """

    name: str
    turns: list[Turn]
    reset: list[dict] = field(default_factory=list)
    judge: dict = field(default_factory=lambda: {"service": "ollama", "model": "qwen2.5:3b"})
    fast: bool = False
    fixtures: dict = field(default_factory=dict)
    source_path: Path | None = None


def load_scenario(path: str | Path) -> Scenario:
    """Parse a scenario YAML file into a :class:`Scenario`.

    Args:
        path: Path to a YAML file with the scenario schema.

    Returns:
        The parsed scenario.

    Raises:
        ValueError: If the file structure is invalid.
        FileNotFoundError: If the path doesn't exist.
    """
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")

    name = data.get("name")
    if not name or not isinstance(name, str):
        raise ValueError(f"{path}: missing or invalid 'name:' field")

    raw_turns = data.get("turns")
    if not isinstance(raw_turns, list):
        raise ValueError(f"{path}: 'turns:' must be a list")

    turns = [_parse_turn(t, path, idx) for idx, t in enumerate(raw_turns)]

    raw_reset = data.get("reset")
    if raw_reset is None:
        reset: list[dict] = []
    elif isinstance(raw_reset, list):
        reset = raw_reset
    else:
        raise ValueError(f"{path}: 'reset:' must be a list of message dicts")

    judge = data.get("judge") or {"service": "ollama", "model": "qwen2.5:3b"}
    if not isinstance(judge, dict):
        raise ValueError(f"{path}: 'judge:' must be a mapping")

    fast = data.get("fast", False)
    if not isinstance(fast, bool):
        raise ValueError(f"{path}: 'fast:' must be a boolean")

    return Scenario(
        name=name,
        turns=turns,
        reset=reset,
        judge=judge,
        fast=fast,
        fixtures=data.get("fixtures") or {},
        source_path=path,
    )


def _parse_turn(t: Any, path: Path, idx: int) -> Turn:
    """Parse one entry from the ``turns:`` list."""
    if not isinstance(t, dict):
        raise ValueError(f"{path}: turn #{idx} must be a mapping")

    user = t.get("user")
    if user is not None and not isinstance(user, str):
        raise ValueError(f"{path}: turn #{idx} 'user:' must be a string if present")

    raw_expect = t.get("expect")
    if not isinstance(raw_expect, list):
        raise ValueError(f"{path}: turn #{idx} missing or invalid 'expect:' list")

    expect = [_parse_expectation(e, path, idx, ei) for ei, e in enumerate(raw_expect)]

    send_after = _parse_send_after(t.get("send_after"), path, idx) if "send_after" in t else None
    if send_after is not None and user is None:
        raise ValueError(
            f"{path}: turn #{idx} has 'send_after:' but no 'user:' — "
            "send_after only schedules when the user message gets sent"
        )

    return Turn(user=user, expect=expect, send_after=send_after)


def _parse_send_after(s: Any, path: Path, turn_idx: int) -> SendAfter:
    """Parse a ``send_after:`` block."""
    if not isinstance(s, dict):
        raise ValueError(f"{path}: turn #{turn_idx} 'send_after:' must be a mapping")

    event = s.get("event")
    if not event or not isinstance(event, str):
        raise ValueError(f"{path}: turn #{turn_idx} 'send_after:' missing or invalid 'event:'")

    delay_ms = s.get("delay_ms", 0)
    if not isinstance(delay_ms, int) or delay_ms < 0:
        raise ValueError(
            f"{path}: turn #{turn_idx} 'send_after.delay_ms' must be a non-negative int"
        )

    return SendAfter(event=event, delay_ms=delay_ms)


def _parse_expectation(e: Any, path: Path, turn_idx: int, exp_idx: int) -> Expectation:
    """Parse one entry from a turn's ``expect:`` list."""
    if not isinstance(e, dict):
        raise ValueError(f"{path}: turn #{turn_idx} expectation #{exp_idx} must be a mapping")

    event = e.get("event")
    if not event or not isinstance(event, str):
        raise ValueError(
            f"{path}: turn #{turn_idx} expectation #{exp_idx} missing or invalid 'event:'"
        )

    criterion = e.get("eval")
    if criterion is not None and event not in JUDGEABLE_EVENTS:
        logger.warning(
            f"{path}: turn #{turn_idx} expectation #{exp_idx}: 'eval:' on "
            f"event {event!r} — judge only makes sense on bot-generated text "
            f"events ({', '.join(sorted(JUDGEABLE_EVENTS))}). Will run but is "
            "unlikely to be meaningful."
        )

    return Expectation(
        event=event,
        within_ms=e.get("within_ms"),
        transcript_contains=e.get("transcript_contains"),
        text_contains=e.get("text_contains"),
        name=e.get("name"),
        args=e.get("args"),
        eval=criterion,
        raw=e,
    )
