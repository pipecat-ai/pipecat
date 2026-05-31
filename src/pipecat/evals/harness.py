#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Scenario harness: drives a bot via the eval transport and asserts on events.

Connects to a running bot's eval transport WebSocket, walks through a parsed
:class:`~pipecat.evals.scenario.Scenario`, and verifies that the expected
semantic events arrive in order with the right payloads and within their
latency budgets.

Matching semantics: expected events must appear in the specified order, but
unmatched events may appear between them (so a scenario doesn't have to
enumerate every event the bot emits). The ``within_ms`` budget for each
expectation is measured from the most recent ``user_input`` send.

A background reader task continuously drains the WebSocket so that
``send_after:`` blocks can anchor on events that fired during prior turns.

Example::

    scenario = load_scenario("scenarios/greeting.yaml")
    result = await run_scenario(scenario, "ws://localhost:7860")
    if result.passed:
        print("PASS")
    else:
        for f in result.failures:
            print(f"  {f}")
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from pipecat.evals.judge import Judge, build_default_judge
from pipecat.evals.scenario import Expectation, Scenario, SendAfter, Turn

# ``websockets`` is imported lazily inside the functions that use it. That keeps
# this module importable (and the CLI plugin loadable via pipecat-cli) even
# when the optional ``websockets-base`` extra isn't installed — users who
# actually run an eval get a clear ImportError at that point.

DEFAULT_EVENT_TIMEOUT_MS = 5000
SEND_AFTER_MAX_WAIT_S = 30.0
SEND_AFTER_POLL_S = 0.01


@dataclass
class AssertionFailure:
    """A single failed assertion within a scenario run."""

    turn_index: int
    expectation_index: int
    event_name: str
    reason: str

    def __str__(self) -> str:
        return (
            f"turn {self.turn_index} expectation {self.expectation_index} "
            f"({self.event_name}): {self.reason}"
        )


@dataclass
class ScenarioResult:
    """Outcome of running a scenario."""

    scenario_name: str
    passed: bool
    failures: list[AssertionFailure] = field(default_factory=list)
    duration_ms: int = 0
    events_seen: list[dict] = field(default_factory=list)


@dataclass
class _RunContext:
    """Per-scenario run state shared across helpers."""

    ws: Any
    queue: asyncio.Queue
    latest_event_times: dict[str, float]
    events_seen: list[dict]


async def run_scenario(
    scenario: Scenario,
    bot_url: str,
    connect_timeout_s: float = 5.0,
) -> ScenarioResult:
    """Run a scenario against a bot at the given WebSocket URL.

    Args:
        scenario: The parsed scenario to run.
        bot_url: WebSocket URL of the bot's eval transport.
        connect_timeout_s: How long to wait for the bot to accept the WS
            connection before giving up.

    Returns:
        The structured outcome.
    """
    import websockets  # lazy: see note at the top of the module

    started = time.monotonic()
    failures: list[AssertionFailure] = []
    events_seen: list[dict] = []

    try:
        async with asyncio.timeout(connect_timeout_s):
            ws = await websockets.connect(bot_url)
    except (TimeoutError, OSError) as e:
        return ScenarioResult(
            scenario_name=scenario.name,
            passed=False,
            failures=[
                AssertionFailure(
                    turn_index=-1,
                    expectation_index=-1,
                    event_name="<connect>",
                    reason=f"failed to connect to {bot_url}: {e.__class__.__name__}",
                )
            ],
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    # Lazily construct the judge — only if the scenario uses eval: assertions.
    judge: Judge | None = None
    needs_judge = any(exp.eval is not None for turn in scenario.turns for exp in turn.expect)
    if needs_judge:
        judge = build_default_judge(scenario.judge)

    ctx = _RunContext(
        ws=ws,
        queue=asyncio.Queue(),
        latest_event_times={},
        events_seen=events_seen,
    )

    reader_task = asyncio.create_task(_reader_loop(ctx))

    try:
        await ws.send(json.dumps({"type": "ready"}))
        # Always send a reset before driving turns — guarantees a clean LLM
        # context per eval. Bots without a context aggregator silently ignore
        # the resulting LLMMessagesUpdateFrame.
        await ws.send(json.dumps({"type": "reset", "messages": scenario.reset}))
        # Push per-eval runtime settings (e.g. ``fast`` to skip audio pacing).
        await ws.send(json.dumps({"type": "settings", "fast": scenario.fast}))

        for turn_idx, turn in enumerate(scenario.turns):
            turn_failures = await _run_turn(ctx, turn, turn_idx, judge)
            failures.extend(turn_failures)
    finally:
        reader_task.cancel()
        try:
            await reader_task
        except (asyncio.CancelledError, Exception):
            pass
        await ws.close()

    return ScenarioResult(
        scenario_name=scenario.name,
        passed=not failures,
        failures=failures,
        duration_ms=int((time.monotonic() - started) * 1000),
        events_seen=events_seen,
    )


async def _reader_loop(ctx: _RunContext):
    """Continuously drain the WS, pushing events into the queue and updating timestamps."""
    import websockets  # lazy: see note at the top of the module

    try:
        async for raw in ctx.ws:
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(f"Harness: dropping non-JSON message: {raw!r}")
                continue
            ctx.events_seen.append(event)
            evt_type = event.get("type", "")
            ctx.latest_event_times[evt_type] = time.monotonic()
            await ctx.queue.put(event)
    except (websockets.ConnectionClosed, asyncio.CancelledError):
        pass


async def _run_turn(
    ctx: _RunContext,
    turn: Turn,
    turn_idx: int,
    judge: Judge | None,
) -> list[AssertionFailure]:
    """Drive one turn: optionally honor send_after, send user_input, match expectations."""
    failures: list[AssertionFailure] = []

    if turn.send_after is not None:
        try:
            await _wait_send_after(ctx, turn.send_after)
        except TimeoutError as e:
            failures.append(
                AssertionFailure(
                    turn_index=turn_idx,
                    expectation_index=-1,
                    event_name=turn.send_after.event,
                    reason=f"send_after never fired: {e}",
                )
            )
            return failures

    anchor = time.monotonic()
    if turn.user is not None:
        await ctx.ws.send(json.dumps({"type": "user_input", "text": turn.user}))
        anchor = time.monotonic()

    for exp_idx, expectation in enumerate(turn.expect):
        budget_ms = expectation.within_ms or DEFAULT_EVENT_TIMEOUT_MS

        try:
            matched = await _await_event(ctx, expectation, anchor, budget_ms)
        except TimeoutError:
            failures.append(
                AssertionFailure(
                    turn_index=turn_idx,
                    expectation_index=exp_idx,
                    event_name=expectation.event,
                    reason=(
                        f"no matching {expectation.event!r} event arrived within {budget_ms}ms"
                    ),
                )
            )
            break

        failure = _check_payload(matched, expectation, turn_idx, exp_idx)
        if failure:
            failures.append(failure)

        judge_failure = await _check_judge(matched, expectation, turn_idx, exp_idx, judge)
        if judge_failure:
            failures.append(judge_failure)

    return failures


async def _wait_send_after(ctx: _RunContext, send_after: SendAfter) -> None:
    """Block until ``send_after.event`` has been seen + ``delay_ms`` has elapsed.

    If the event was seen earlier in the run, anchor on that time (potentially
    fire immediately). Otherwise, poll the latest_event_times map until the
    event arrives, then anchor on that.
    """
    target_delay_s = send_after.delay_ms / 1000.0
    deadline = time.monotonic() + SEND_AFTER_MAX_WAIT_S

    while True:
        seen_at = ctx.latest_event_times.get(send_after.event)
        if seen_at is not None:
            wait_s = max(0.0, (seen_at + target_delay_s) - time.monotonic())
            await asyncio.sleep(wait_s)
            return

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"event {send_after.event!r} not seen within {int(SEND_AFTER_MAX_WAIT_S * 1000)}ms"
            )

        await asyncio.sleep(SEND_AFTER_POLL_S)


async def _await_event(
    ctx: _RunContext,
    expectation: Expectation,
    anchor: float,
    budget_ms: int,
) -> dict:
    """Pop events from the queue until one matching ``expectation.event`` arrives.

    Events that don't match the expected name are dropped (so a scenario doesn't
    have to enumerate every event the bot emits). They remain in ``events_seen``
    and ``latest_event_times`` for diagnostics and send_after lookups.
    """
    deadline = anchor + (budget_ms / 1000.0)

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError()

        try:
            async with asyncio.timeout(remaining):
                event = await ctx.queue.get()
        except TimeoutError:
            raise

        if event.get("type") == expectation.event:
            return event


def _check_payload(
    event: dict,
    expectation: Expectation,
    turn_idx: int,
    exp_idx: int,
) -> AssertionFailure | None:
    """Apply payload-level checks to a matched event. Returns the first failure or None."""

    def fail(reason: str) -> AssertionFailure:
        return AssertionFailure(
            turn_index=turn_idx,
            expectation_index=exp_idx,
            event_name=expectation.event,
            reason=reason,
        )

    if expectation.transcript_contains is not None:
        transcript = event.get("transcript", "")
        if expectation.transcript_contains not in transcript:
            return fail(
                f"transcript {transcript!r} does not contain {expectation.transcript_contains!r}"
            )

    if expectation.text_contains is not None:
        text = event.get("text", "")
        if expectation.text_contains not in text:
            return fail(f"text {text!r} does not contain {expectation.text_contains!r}")

    if expectation.name is not None:
        actual_name = event.get("name")
        if actual_name != expectation.name:
            return fail(f"name {actual_name!r} != expected {expectation.name!r}")

    if expectation.args is not None:
        actual_args = event.get("args")
        if actual_args != expectation.args:
            return fail(f"args {actual_args!r} != expected {expectation.args!r}")

    return None


async def _check_judge(
    event: dict,
    expectation: Expectation,
    turn_idx: int,
    exp_idx: int,
    judge: Judge | None,
) -> AssertionFailure | None:
    """Run the judge assertion if ``eval:`` was set on this expectation."""
    if expectation.eval is None:
        return None

    if judge is None:
        return AssertionFailure(
            turn_index=turn_idx,
            expectation_index=exp_idx,
            event_name=expectation.event,
            reason="scenario uses 'eval:' but no judge could be built",
        )

    content = event.get("text") or event.get("transcript")
    if not content:
        return AssertionFailure(
            turn_index=turn_idx,
            expectation_index=exp_idx,
            event_name=expectation.event,
            reason=f"event has no text/transcript to judge: {event!r}",
        )

    verdict = await judge.evaluate(expectation.eval, content)
    if not verdict.passed:
        return AssertionFailure(
            turn_index=turn_idx,
            expectation_index=exp_idx,
            event_name=expectation.event,
            reason=f"eval {expectation.eval!r}: judge said no — {verdict.reason}",
        )

    return None
