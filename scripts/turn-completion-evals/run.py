#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Turn-completion marker eval: how well LLMs follow the ●/◐/○ protocol.

Drives Pipecat's real LLM services, with ``filter_incomplete_user_turns``
enabled and no audio or transport around them, through scripted
conversations and scores what
:class:`~pipecat.turns.user_turn_completion_mixin.UserTurnCompletionLLMServiceMixin`
sees: which marker the model chose, whether it came first, whether a ● carried
speech, whether a ◐/○ came alone. See ``README.md`` for the case and model file
formats and the failure kinds.

Usage::

    uv run python scripts/turn-completion-evals/run.py                 # every model, every case
    uv run python scripts/turn-completion-evals/run.py -m openai -m groq
    uv run python scripts/turn-completion-evals/run.py -c cutoff --repeat 3
    uv run python scripts/turn-completion-evals/run.py --prompt terse --mode live
"""

import argparse
import asyncio
import fnmatch
import importlib
import json
import os
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from contextlib import AbstractAsyncContextManager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import yaml
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    FunctionCallsStartedFrame,
    LLMContextFrame,
    LLMMarkerFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.settings import LLMSettings
from pipecat.tests.utils import QueuedFrameProcessor
from pipecat.turns.user_turn_completion_mixin import (
    USER_TURN_COMPLETE_MARKER,
    USER_TURN_INCOMPLETE_LONG_MARKER,
    USER_TURN_INCOMPLETE_SHORT_MARKER,
    UserTurnCompletionConfig,
    _render_incomplete_long_prompt,
    _render_incomplete_short_prompt,
)
from pipecat.workers.runner import WorkerRunner

HERE = Path(__file__).resolve().parent

# A case whose history opens with the bot gets this first, the way a bot's
# on-connect handler kicks off the first line. Neutral on purpose: it must
# not tell the model what that line was. Providers that require a user-first
# conversation (Bedrock) convert it to a user message.
KICKOFF_INSTRUCTION = "The user has joined the conversation."

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant in a voice conversation. Your responses will be spoken "
    "aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. "
    "Respond to what the user said in a creative, helpful, and brief way."
)

# Categories whose user turns also run as an STT-style transcript (lowercase,
# no punctuation), since that is what a real bot's LLM sees.
STT_VARIANT_CATEGORIES = {
    "complete",
    "short_complete",
    "cutoff",
    "time_request",
    "preamble",
    "tools",
    "hard",
    "long",
    "continuation",
}

# Failure kinds. Format kinds describe the shape of the raw output; verdict
# kinds compare the framework's reading of it with the case's expectation.
FORMAT_KINDS = (
    "no_marker",
    "marker_not_first",
    "multiple_markers",
    "bare_complete",
    "text_after_incomplete",
)
VERDICT_KINDS = (
    "false_complete",  # expected ◐/○, got ●
    "false_incomplete",  # expected ●, got ◐/○
    "missing_tool_call",
    "unexpected_tool_call",
    "wrong_incomplete_type",  # soft: ◐ vs ○
    "judge_no",
    "error",
    "timeout",
)


# ---------------------------------------------------------------------------
# Tools a case can advertise
# ---------------------------------------------------------------------------


async def _tool_handler(params: FunctionCallParams):
    await params.result_callback({"ok": True, "note": "stubbed for the eval"})


TOOLS = {
    "get_current_weather": FunctionSchema(
        name="get_current_weather",
        description="Get the current weather for a location.",
        properties={"location": {"type": "string", "description": "City and state or country."}},
        required=["location"],
        handler=_tool_handler,
    ),
    "book_appointment": FunctionSchema(
        name="book_appointment",
        description="Book an appointment on a given day and time.",
        properties={
            "day": {"type": "string", "description": "Day of the appointment."},
            "time": {"type": "string", "description": "Time of the appointment."},
        },
        required=["day", "time"],
        handler=_tool_handler,
    ),
}


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    role: str  # user | bot | developer
    text: str
    expect: str | None = None  # complete | short | long | incomplete | tool:<name>
    eval: str | None = None
    infer: bool = False


@dataclass
class Case:
    name: str
    category: str
    turns: list[Turn]
    tools: list[str] = field(default_factory=list)
    system: str | None = None
    variant: str = "punctuated"


def _stt_style(text: str) -> str:
    """Lowercase, no punctuation, single spaces: what a streaming STT hands the LLM."""
    text = text.replace("...", " ").replace("…", " ")
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch == "'" or not cat.startswith("P"):
            out.append(ch.lower())
        else:
            out.append(" ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _remap_markers(text: str, markers: tuple[str, str, str]) -> str:
    """Case files are written with the default markers; swap in the configured ones."""
    defaults = (
        USER_TURN_COMPLETE_MARKER,
        USER_TURN_INCOMPLETE_SHORT_MARKER,
        USER_TURN_INCOMPLETE_LONG_MARKER,
    )
    for default, configured in zip(defaults, markers):
        text = text.replace(default, configured)
    return text


def _developer_macro(text: str, markers: tuple[str, str, str]) -> str:
    if text == "$reprompt_short":
        return _render_incomplete_short_prompt(*markers)
    if text == "$reprompt_long":
        return _render_incomplete_long_prompt(*markers)
    return text


def load_cases(cases_dir: Path, patterns: list[str], markers: tuple[str, str, str]) -> list[Case]:
    cases: list[Case] = []
    for path in sorted(cases_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        category = data["category"]
        for raw in data["cases"]:
            turns = []
            for t in raw["turns"]:
                if "user" in t:
                    # A user turn without `expect` is history only, no inference.
                    turns.append(
                        Turn("user", t["user"], t.get("expect"), t.get("eval"), infer="expect" in t)
                    )
                elif "developer" in t:
                    turns.append(
                        Turn(
                            "developer",
                            _developer_macro(t["developer"], markers),
                            t.get("expect"),
                            t.get("eval"),
                            infer="expect" in t,
                        )
                    )
                elif "bot" in t:
                    turns.append(Turn("bot", _remap_markers(t["bot"], markers)))
                else:
                    raise ValueError(f"{path}: turn needs user/bot/developer: {t}")
            for t in turns:
                if t.infer and not t.expect:
                    raise ValueError(f"{path}:{raw['name']}: inference turn needs expect")
            case = Case(raw["name"], category, turns, raw.get("tools", []), raw.get("system"))
            cases.append(case)
            stt = raw.get("stt_variant", category in STT_VARIANT_CATEGORIES)
            if stt:
                stt_turns = [
                    Turn(
                        t.role,
                        _stt_style(t.text) if t.role == "user" else t.text,
                        t.expect,
                        t.eval,
                        t.infer,
                    )
                    for t in turns
                ]
                cases.append(
                    Case(f"{raw['name']}__stt", category, stt_turns, case.tools, case.system, "stt")
                )
    if patterns:
        cases = [c for c in cases if any(p in c.name or p in c.category for p in patterns)]
    return cases


def _context_text(role: str, text: str, markers: tuple[str, str, str]) -> str:
    """The message content as the assistant aggregator would have stored it."""
    if role != "bot":
        return text
    if text.strip() in markers or any(text.startswith(m) for m in markers):
        return text
    return f"{markers[0]} {text}"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    provider: str
    model: str
    service: str  # dotted class path
    kwargs: dict[str, Any]
    settings: dict[str, Any]
    system_suffix: str | None
    concurrency: int
    label: str
    tools: bool = True
    developer_role: bool = True

    @property
    def id(self) -> str:
        return self.label


def _env_subst(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        name = value[1:]
        if name not in os.environ:
            raise KeyError(name)
        return os.environ[name]
    if isinstance(value, dict):
        return {k: _env_subst(v) for k, v in value.items()}
    return value


def load_models(
    path: Path, patterns: list[str], excludes: list[str] | None = None
) -> tuple[list[ModelSpec], list[str]]:
    data = yaml.safe_load(path.read_text())
    specs: list[ModelSpec] = []
    skipped: list[str] = []
    for provider, pcfg in data["providers"].items():
        try:
            kwargs = _env_subst(pcfg.get("kwargs", {}))
        except KeyError as e:
            skipped.append(f"{provider}: ${e.args[0]} not set")
            continue
        for m in pcfg["models"]:
            model = _env_subst(m["model"])
            label = m.get("label") or f"{provider}/{model}"
            specs.append(
                ModelSpec(
                    provider=provider,
                    model=model,
                    service=pcfg["service"],
                    kwargs=kwargs,
                    settings=m.get("settings", {}),
                    system_suffix=m.get("system_suffix", pcfg.get("system_suffix")),
                    concurrency=int(pcfg.get("concurrency", 4)),
                    label=label,
                    tools=bool(m.get("tools", pcfg.get("tools", True))),
                    developer_role=bool(m.get("developer_role", pcfg.get("developer_role", True))),
                )
            )
    if patterns:
        specs = [s for s in specs if any(fnmatch.fnmatch(s.id, p) or p in s.id for p in patterns)]
    if excludes:
        specs = [
            s for s in specs if not any(fnmatch.fnmatch(s.id, p) or p in s.id for p in excludes)
        ]
    return specs, skipped


def _convert_nested_settings(cls: type, settings: dict[str, Any]) -> dict[str, Any]:
    """Turn the mapping forms of provider config objects into their classes.

    Google's and Anthropic's Settings already accept a mapping for ``thinking``;
    the OpenAI Responses ``reasoning`` and DeepSeek ``thinking`` fields do not.
    """
    out = dict(settings)
    if "reasoning" in out and isinstance(out["reasoning"], dict):
        from pipecat.services.openai.responses.llm import OpenAIResponsesReasoningConfig

        out["reasoning"] = OpenAIResponsesReasoningConfig(**out["reasoning"])
    if "thinking" in out and isinstance(out["thinking"], dict) and "DeepSeek" in cls.__name__:
        from pipecat.services.deepseek.llm import DeepSeekThinkingConfig

        out["thinking"] = DeepSeekThinkingConfig(**out["thinking"])
    return out


def build_service(spec: ModelSpec, system_instruction: str):
    module_name, _, attr = spec.service.rpartition(".")
    cls = getattr(importlib.import_module(module_name), attr)
    if spec.system_suffix:
        system_instruction = f"{system_instruction}\n\n{spec.system_suffix}"
    fields = _convert_nested_settings(cls, spec.settings)
    # `extra` is the pass-through bag of provider request parameters; keep it
    # out of from_mapping, which would nest it as an unknown field.
    extra = fields.pop("extra", {})
    settings = cls.Settings.from_mapping(
        {"model": spec.model, "system_instruction": system_instruction, **fields}
    )
    if extra:
        settings.extra = {**(settings.extra or {}), **extra}
    llm = cls(settings=settings, **spec.kwargs)
    if not spec.developer_role:
        # A model behind a generic OpenAI-compatible service whose chat template
        # rejects the developer role; the adapter then sends those as user.
        llm.supports_developer_role = False
    return llm


# ---------------------------------------------------------------------------
# Driving one inference
# ---------------------------------------------------------------------------


@dataclass
class Inference:
    raw: str
    chunks: list[tuple[float, str]]
    marker_frames: list[str]
    spoken: str
    function_calls: list[dict[str, Any]]
    errors: list[str]
    context_at_ms: float | None
    started_ms: float


async def run_inference(
    spec: ModelSpec,
    context: LLMContext,
    system_instruction: str,
    completion_config: UserTurnCompletionConfig,
    timeout: float,
) -> Inference:
    llm = build_service(spec, system_instruction)

    chunks: list[tuple[float, str]] = []
    times: dict[str, float] = {}

    orig_push_turn_text = llm._push_turn_text

    async def hooked_push_turn_text(text: str):
        chunks.append((time.monotonic(), text))
        await orig_push_turn_text(text)

    llm._push_turn_text = hooked_push_turn_text  # type: ignore[method-assign]

    orig_process_frame = llm.process_frame

    async def hooked_process_frame(frame, direction):
        if isinstance(frame, LLMContextFrame):
            times["context"] = time.monotonic()
        await orig_process_frame(frame, direction)

    llm.process_frame = hooked_process_frame  # type: ignore[method-assign]

    received_up: asyncio.Queue = asyncio.Queue()
    received_down: asyncio.Queue = asyncio.Queue()
    source = QueuedFrameProcessor(
        queue=received_up, queue_direction=FrameDirection.UPSTREAM, ignore_start=True
    )
    sink = QueuedFrameProcessor(
        queue=received_down, queue_direction=FrameDirection.DOWNSTREAM, ignore_start=True
    )
    pipeline = Pipeline([source, llm, sink])
    worker = PipelineWorker(pipeline, cancel_on_idle_timeout=False)
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def _on_started(worker, frame):
        started.set()

    frames = [
        LLMUpdateSettingsFrame(
            delta=LLMSettings(
                filter_incomplete_user_turns=True,
                user_turn_completion_config=completion_config,
            )
        ),
        LLMContextFrame(context),
    ]

    async def push():
        await asyncio.wait_for(started.wait(), timeout=10)
        for frame in frames:
            await worker.queue_frame(frame)
        await worker.queue_frame(EndFrame())

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    t_start = time.monotonic()
    try:
        await asyncio.wait_for(asyncio.gather(runner.run(), push()), timeout=timeout)
    except TimeoutError:
        await runner.cancel()
        raise

    marker_frames: list[str] = []
    spoken: list[str] = []
    function_calls: list[dict[str, Any]] = []
    while not received_down.empty():
        frame = received_down.get_nowait()
        if isinstance(frame, LLMMarkerFrame):
            marker_frames.append(frame.marker)
        elif isinstance(frame, LLMTextFrame):
            spoken.append(frame.text)
        elif isinstance(frame, FunctionCallsStartedFrame):
            for call in frame.function_calls:
                function_calls.append(
                    {"name": call.function_name, "arguments": dict(call.arguments)}
                )
    errors: list[str] = []
    while not received_up.empty():
        frame = received_up.get_nowait()
        if isinstance(frame, ErrorFrame):
            errors.append(str(frame.error))

    return Inference(
        raw="".join(t for _, t in chunks),
        chunks=chunks,
        marker_frames=marker_frames,
        spoken="".join(spoken),
        function_calls=function_calls,
        errors=errors,
        context_at_ms=times.get("context"),
        started_ms=t_start,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    model: str
    provider: str
    case: str
    category: str
    variant: str
    attempt: int
    step: int
    role: str
    input: str
    expect: str
    verdict: str  # complete | short | long | none
    raw: str
    spoken: str
    function_calls: list[dict[str, Any]]
    kinds: list[str]
    passed: bool
    verdict_ok: bool
    format_ok: bool
    timing_ok: bool | None
    judge: dict[str, Any] | None
    ttfb_ms: float | None
    marker_ms: float | None
    chars_before_marker: int | None
    error: str | None
    duration_ms: float


def _framework_verdict(marker_frames: list[str], markers: tuple[str, str, str]) -> str:
    complete, short, long = markers
    for m in marker_frames:
        if m == short:
            return "short"
        if m == long:
            return "long"
        if m == complete:
            return "complete"
    return "none"


def score(
    turn: Turn, inf: Inference, markers: tuple[str, str, str]
) -> tuple[str, list[str], bool, bool, bool | None, dict[str, Any]]:
    complete, short, long = markers
    kinds: list[str] = []
    raw = inf.raw.lstrip()

    positions = {m: raw.find(m) for m in markers if raw.find(m) >= 0}
    if not positions:
        kinds.append("no_marker")
        first_marker = None
        first_pos = None
    else:
        first_marker, first_pos = min(positions.items(), key=lambda kv: kv[1])
        if first_pos > 0:
            kinds.append("marker_not_first")
        if sum(raw.count(m) for m in markers) > 1:
            kinds.append("multiple_markers")
        after = raw[first_pos + len(first_marker) :].strip()
        if first_marker == complete and not after and not inf.function_calls:
            kinds.append("bare_complete")
        if first_marker in (short, long) and after:
            kinds.append("text_after_incomplete")

    verdict = _framework_verdict(inf.marker_frames, markers)
    expect = turn.expect or ""
    tool_names = [c["name"] for c in inf.function_calls]
    timing_ok: bool | None = None

    if expect.startswith("tool:"):
        wanted = expect.split(":", 1)[1]
        if wanted not in tool_names:
            kinds.append("missing_tool_call")
        if verdict in ("short", "long"):
            kinds.append("false_incomplete")
        # A tool call is a commitment to the turn, so ● may be absent.
        kinds = [k for k in kinds if k != "no_marker"]
    elif expect == "complete":
        if verdict in ("short", "long"):
            kinds.append("false_incomplete")
        elif verdict == "none" and not inf.function_calls:
            pass  # already no_marker
    elif expect in ("short", "long", "incomplete"):
        if verdict == "complete":
            kinds.append("false_complete")
        elif verdict in ("short", "long") and expect != "incomplete":
            timing_ok = verdict == expect
            if not timing_ok:
                kinds.append("wrong_incomplete_type")
        if tool_names:
            kinds.append("unexpected_tool_call")
    else:
        raise ValueError(f"unknown expect: {expect!r}")

    hard_verdict_kinds = {
        "false_complete",
        "false_incomplete",
        "missing_tool_call",
        "unexpected_tool_call",
        "no_marker",
    }
    verdict_ok = not (set(kinds) & hard_verdict_kinds)
    format_ok = not (set(kinds) & set(FORMAT_KINDS))

    # Latency
    metrics: dict[str, Any] = {"ttfb_ms": None, "marker_ms": None, "chars_before_marker": None}
    if inf.chunks and inf.context_at_ms is not None:
        metrics["ttfb_ms"] = round((inf.chunks[0][0] - inf.context_at_ms) * 1000)
        if first_marker is not None:
            acc = ""
            for t, text in inf.chunks:
                acc += text
                if first_marker in acc:
                    metrics["marker_ms"] = round((t - inf.context_at_ms) * 1000)
                    break
            metrics["chars_before_marker"] = len(inf.raw[: inf.raw.find(first_marker)].strip())
    return verdict, kinds, verdict_ok, format_ok, timing_ok, metrics


# ---------------------------------------------------------------------------
# Judge (optional, local Ollama by default)
# ---------------------------------------------------------------------------


class Judge:
    def __init__(self, config: dict[str, Any] | None):
        from pipecat.evals.judge import EvalJudge

        self._config = config
        self._cls = EvalJudge

    async def evaluate(self, history: list[tuple[str, str]], reply: str, criterion: str) -> dict:
        judge = self._cls.from_config(self._config)
        for role, text in history:
            if role == "user":
                judge.add_user_message(text)
            elif role == "bot":
                judge.add_assistant_message(text)
        judge.add_assistant_message(reply)
        verdict = await judge.evaluate(criterion)
        return {"passed": verdict.passed, "verdict": verdict.verdict, "reason": verdict.reason}


# ---------------------------------------------------------------------------
# Running cases
# ---------------------------------------------------------------------------


async def run_case(
    spec: ModelSpec,
    case: Case,
    attempt: int,
    *,
    markers: tuple[str, str, str],
    completion_config: UserTurnCompletionConfig,
    mode: str,
    timeout: float,
    judge: Judge | None,
    sem: AbstractAsyncContextManager,
) -> list[StepResult]:
    results: list[StepResult] = []
    messages: list[dict[str, Any]] = []
    history: list[tuple[str, str]] = []  # for the judge, marker-free
    system_instruction = case.system or DEFAULT_SYSTEM_INSTRUCTION
    tools = ToolsSchema(standard_tools=[TOOLS[n] for n in case.tools]) if case.tools else None
    if case.turns and case.turns[0].role == "bot":
        messages.append({"role": "developer", "content": KICKOFF_INSTRUCTION})

    for i, turn in enumerate(case.turns):
        role = "assistant" if turn.role == "bot" else turn.role
        if not turn.infer:
            messages.append({"role": role, "content": _context_text(turn.role, turn.text, markers)})
            if turn.role == "bot":
                stripped = turn.text
                for m in markers:
                    stripped = stripped.replace(m, "")
                if stripped.strip():
                    history.append(("bot", stripped.strip()))
            elif turn.role == "user":
                history.append(("user", turn.text))
            continue

        messages.append({"role": role, "content": turn.text})
        if turn.role == "user":
            history.append(("user", turn.text))
        ctx_messages = cast(list[LLMContextMessage], list(messages))
        context = (
            LLMContext(messages=ctx_messages, tools=tools)
            if tools
            else LLMContext(messages=ctx_messages)
        )

        t0 = time.monotonic()
        error: str | None = None
        inf: Inference | None = None
        async with sem:
            try:
                inf = await run_inference(
                    spec, context, system_instruction, completion_config, timeout
                )
                if inf.errors and not inf.raw and not inf.function_calls:
                    error = inf.errors[0]
            except TimeoutError:
                error = "timeout"
            except Exception as e:  # provider/client construction errors
                error = f"{type(e).__name__}: {e}"
        duration_ms = round((time.monotonic() - t0) * 1000)

        if inf is None or error:
            kind = "timeout" if error == "timeout" else "error"
            results.append(
                StepResult(
                    model=spec.id,
                    provider=spec.provider,
                    case=case.name,
                    category=case.category,
                    variant=case.variant,
                    attempt=attempt,
                    step=i,
                    role=turn.role,
                    input=turn.text,
                    expect=turn.expect or "",
                    verdict="none",
                    raw=inf.raw if inf else "",
                    spoken="",
                    function_calls=[],
                    kinds=[kind],
                    passed=False,
                    verdict_ok=False,
                    format_ok=False,
                    timing_ok=None,
                    judge=None,
                    ttfb_ms=None,
                    marker_ms=None,
                    chars_before_marker=None,
                    error=error,
                    duration_ms=duration_ms,
                )
            )
            break  # the rest of the case depends on this step

        verdict, kinds, verdict_ok, format_ok, timing_ok, metrics = score(turn, inf, markers)

        judge_result = None
        if judge and turn.eval and verdict == "complete" and inf.spoken.strip():
            try:
                judge_result = await judge.evaluate(history, inf.spoken.strip(), turn.eval)
                if not judge_result["passed"]:
                    kinds.append("judge_no")
            except Exception as e:
                judge_result = {"passed": None, "verdict": "error", "reason": str(e)}

        passed = verdict_ok and format_ok and "judge_no" not in kinds
        results.append(
            StepResult(
                model=spec.id,
                provider=spec.provider,
                case=case.name,
                category=case.category,
                variant=case.variant,
                attempt=attempt,
                step=i,
                role=turn.role,
                input=turn.text,
                expect=turn.expect or "",
                verdict=verdict,
                raw=inf.raw,
                spoken=inf.spoken,
                function_calls=inf.function_calls,
                kinds=kinds,
                passed=passed,
                verdict_ok=verdict_ok,
                format_ok=format_ok,
                timing_ok=timing_ok,
                judge=judge_result,
                ttfb_ms=metrics["ttfb_ms"],
                marker_ms=metrics["marker_ms"],
                chars_before_marker=metrics["chars_before_marker"],
                error=None,
                duration_ms=duration_ms,
            )
        )

        # What the assistant aggregator would have written to the context.
        if mode == "live":
            if verdict == "complete":
                stored = f"{markers[0]} {inf.spoken.strip()}".strip()
                history.append(("bot", inf.spoken.strip()))
            elif verdict == "short":
                stored = markers[1]
            elif verdict == "long":
                stored = markers[2]
            else:
                stored = inf.raw.strip()
                if stored:
                    history.append(("bot", stored))
            if stored:
                messages.append({"role": "assistant", "content": stored})
        else:
            canonical = _canonical_reply(turn, markers)
            if canonical:
                messages.append({"role": "assistant", "content": canonical})
                if canonical not in markers:
                    history.append(("bot", canonical[len(markers[0]) :].strip()))
    return results


def _canonical_reply(turn: Turn, markers: tuple[str, str, str]) -> str | None:
    """The assistant entry a well-behaved model would leave after this step.

    ◐/○ are stored bare; a ● step has no canonical text, so the case's next
    ``bot:`` turn (if any) supplies it.
    """
    if turn.expect == "short":
        return markers[1]
    if turn.expect in ("long", "incomplete"):
        return markers[2]
    return None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:5.1f}%" if d else "  n/a"


def _median(values: list[float | None]) -> str:
    vals = [v for v in values if v is not None]
    return f"{statistics.median(vals):.0f}" if vals else "-"


def write_summary(out_dir: Path, results: list[StepResult], args: argparse.Namespace) -> str:
    by_model: dict[str, list[StepResult]] = defaultdict(list)
    for r in results:
        by_model[r.model].append(r)

    lines = [f"# Turn-completion marker eval — {out_dir.name}", ""]
    lines.append(
        f"prompt: `{args.prompt or 'default'}`  markers: `{''.join(args.markers)}`  "
        f"mode: `{args.mode}`  repeat: {args.repeat}  judge: {'on' if args.judge else 'off'}"
    )
    lines.append("")
    lines.append("## Leaderboard")
    lines.append("")
    lines.append(
        "| model | steps | pass | verdict | format | ◐/○ type | false ● | false ◐○ | bare ● | "
        "text after ◐○ | no marker | not first | errors | ttfb ms | marker ms |"
    )
    lines.append("|" + "---|" * 15)
    rows = []
    for model, rs in by_model.items():
        n = len(rs)
        scored = [r for r in rs if not r.error]
        kinds = Counter(k for r in rs for k in r.kinds)
        timed = [r for r in scored if r.timing_ok is not None]
        rows.append(
            (
                sum(r.passed for r in rs) / n if n else 0,
                f"| {model} | {n} | {_pct(sum(r.passed for r in rs), n)} | "
                f"{_pct(sum(r.verdict_ok for r in scored), len(scored))} | "
                f"{_pct(sum(r.format_ok for r in scored), len(scored))} | "
                f"{_pct(sum(bool(r.timing_ok) for r in timed), len(timed))} | "
                f"{kinds['false_complete']} | {kinds['false_incomplete']} | {kinds['bare_complete']} | "
                f"{kinds['text_after_incomplete']} | {kinds['no_marker']} | {kinds['marker_not_first']} | "
                f"{kinds['error'] + kinds['timeout']} | {_median([r.ttfb_ms for r in scored])} | "
                f"{_median([r.marker_ms for r in scored])} |",
            )
        )
    for _, row in sorted(rows, key=lambda x: -x[0]):
        lines.append(row)

    # Category matrix
    categories = sorted({r.category for r in results})
    lines += ["", "## Pass rate by category", ""]
    lines.append("| model | " + " | ".join(categories) + " |")
    lines.append("|" + "---|" * (len(categories) + 1))
    for model, rs in sorted(
        by_model.items(), key=lambda kv: -sum(r.passed for r in kv[1]) / max(len(kv[1]), 1)
    ):
        cells = []
        for cat in categories:
            sub = [r for r in rs if r.category == cat]
            cells.append(_pct(sum(r.passed for r in sub), len(sub)).strip())
        lines.append(f"| {model} | " + " | ".join(cells) + " |")

    # Variant comparison
    variants = sorted({r.variant for r in results})
    if len(variants) > 1:
        lines += ["", "## Punctuated vs STT-style input", ""]
        lines.append("| model | " + " | ".join(variants) + " |")
        lines.append("|" + "---|" * (len(variants) + 1))
        for model, rs in by_model.items():
            cells = []
            for v in variants:
                sub = [r for r in rs if r.variant == v]
                cells.append(_pct(sum(r.passed for r in sub), len(sub)).strip())
            lines.append(f"| {model} | " + " | ".join(cells) + " |")

    # Hardest steps
    by_step: dict[tuple[str, int, str], list[StepResult]] = defaultdict(list)
    for r in results:
        if not r.error:
            by_step[(r.case, r.step, r.expect)].append(r)
    hardest = sorted(by_step.items(), key=lambda kv: sum(not r.passed for r in kv[1]), reverse=True)
    lines += ["", "## Hardest steps (fails across models)", ""]
    lines.append("| case | step | expect | fails | of | top failure kinds |")
    lines.append("|---|---|---|---|---|---|")
    for (case, step, expect), rs in hardest[:25]:
        fails = sum(not r.passed for r in rs)
        if not fails:
            break
        kinds = Counter(k for r in rs for k in r.kinds).most_common(3)
        lines.append(
            f"| {case} | {step} | {expect} | {fails} | {len(rs)} | "
            + ", ".join(f"{k} ({n})" for k, n in kinds)
            + " |"
        )

    # Errors
    errored = [r for r in results if r.error]
    if errored:
        lines += ["", "## Errors", ""]
        seen: dict[str, str] = {}
        for r in errored:
            seen.setdefault(r.model, (r.error or "")[:200])
        for model, err in seen.items():
            lines.append(f"- **{model}**: `{err}`")

    text = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(text)
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "-m", "--model", action="append", default=[], help="model id substring/glob (repeatable)"
    )
    p.add_argument(
        "-x",
        "--exclude",
        action="append",
        default=[],
        help="model id substring/glob to skip (repeatable)",
    )
    p.add_argument(
        "-c",
        "--case",
        action="append",
        default=[],
        help="case name/category substring (repeatable)",
    )
    p.add_argument("--models-file", default=str(HERE / "models.yaml"))
    p.add_argument("--cases-dir", default=str(HERE / "cases"))
    p.add_argument(
        "--prompt", help="instructions variant: prompts/<name>.txt (default: the framework's)"
    )
    p.add_argument("--markers", default=None, help="three marker characters, e.g. '●◐○' or '✓…?'")
    p.add_argument("--mode", choices=["canonical", "live"], default="canonical")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("-j", "--concurrency", type=int, default=16, help="global concurrent inferences")
    p.add_argument("-t", "--timeout", type=float, default=90.0)
    p.add_argument("-n", "--name", help="run name (default: timestamp)")
    p.add_argument("--runs-dir", default=str(HERE / "runs"))
    p.add_argument("--judge", action="store_true", help="judge `eval:` criteria with local Ollama")
    p.add_argument("--judge-model", default="gemma4:12b")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--list-cases", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--summarize", metavar="RUN_DIR", help="rebuild summary.md from a run's jsonl files"
    )
    return p.parse_args(argv)


def summarize_dir(run_dir: Path, args: argparse.Namespace) -> str:
    results: list[StepResult] = []
    for path in sorted(run_dir.glob("*.jsonl")):
        with path.open() as f:
            results.extend(StepResult(**json.loads(line)) for line in f if line.strip())
    config = (
        json.loads((run_dir / "config.json").read_text())
        if (run_dir / "config.json").exists()
        else {}
    )
    args.prompt = config.get("prompt")
    args.markers = tuple(config.get("markers", args.markers or "●◐○"))
    args.mode = config.get("mode", "canonical")
    args.repeat = config.get("repeat", 1)
    return write_summary(run_dir, results, args)


async def main_async(args: argparse.Namespace) -> int:
    if args.summarize:
        print(summarize_dir(Path(args.summarize), args))
        return 0
    load_dotenv(HERE.parent.parent / ".env", override=True)
    # Developer setup hooks (e.g. a debugger UI) have no place in a sweep.
    os.environ.pop("PIPECAT_SETUP_FILES", None)
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "ERROR")

    if args.markers:
        chars = [c for c in args.markers if not c.isspace() and c != ","]
        if len(chars) != 3:
            raise SystemExit("--markers needs exactly three characters")
        args.markers = tuple(chars)
    else:
        args.markers = (
            USER_TURN_COMPLETE_MARKER,
            USER_TURN_INCOMPLETE_SHORT_MARKER,
            USER_TURN_INCOMPLETE_LONG_MARKER,
        )
    markers = args.markers

    instructions = None
    if args.prompt:
        template = (HERE / "prompts" / f"{args.prompt}.txt").read_text()
        instructions = template.format(complete=markers[0], short=markers[1], long=markers[2])
    completion_config = UserTurnCompletionConfig(
        instructions=instructions,
        complete_marker=markers[0],
        incomplete_short_marker=markers[1],
        incomplete_long_marker=markers[2],
        # Never let a ◐/○ fire its re-prompt inside the harness.
        incomplete_short_timeout=3600.0,
        incomplete_long_timeout=3600.0,
    )

    specs, skipped = load_models(Path(args.models_file), args.model, args.exclude)
    cases = load_cases(Path(args.cases_dir), args.case, markers)

    if args.list_models:
        for s in specs:
            print(s.id)
        for s in skipped:
            print(f"(skipped) {s}")
        return 0
    if args.list_cases:
        for c in cases:
            steps = sum(t.infer for t in c.turns)
            print(f"{c.category:16} {c.name:48} {steps} step(s)")
        print(f"{len(cases)} cases, {sum(sum(t.infer for t in c.turns) for c in cases)} steps")
        return 0
    if not specs:
        raise SystemExit("no models selected")
    if not cases:
        raise SystemExit("no cases selected")

    judge = (
        Judge(
            {"service": "ollama", "model": args.judge_model, "extra": {"reasoning_effort": "none"}}
        )
        if args.judge
        else None
    )

    out_dir = Path(args.runs_dir) / (args.name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "prompt": args.prompt,
                "instructions": completion_config.completion_instructions,
                "markers": markers,
                "mode": args.mode,
                "repeat": args.repeat,
                "models": [s.id for s in specs],
                "cases": [c.name for c in cases],
                "skipped_providers": skipped,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    total_steps = sum(sum(t.infer for t in c.turns) for c in cases) * len(specs) * args.repeat
    print(
        f"{len(specs)} models x {len(cases)} cases x {args.repeat} = ~{total_steps} inferences "
        f"-> {out_dir}"
    )
    for s in skipped:
        print(f"  skipped {s}")

    global_sem = asyncio.Semaphore(args.concurrency)
    provider_sems: dict[str, asyncio.Semaphore] = {}
    results: list[StepResult] = []
    done = 0
    lock = asyncio.Lock()

    class _Both:
        def __init__(self, a, b):
            self.a, self.b = a, b

        async def __aenter__(self):
            await self.a.__aenter__()
            await self.b.__aenter__()

        async def __aexit__(self, *exc):
            await self.b.__aexit__(*exc)
            await self.a.__aexit__(*exc)

    async def one(spec: ModelSpec, case: Case, attempt: int):
        nonlocal done
        sem = provider_sems.setdefault(spec.provider, asyncio.Semaphore(spec.concurrency))
        rs = await run_case(
            spec,
            case,
            attempt,
            markers=markers,
            completion_config=completion_config,
            mode=args.mode,
            timeout=args.timeout,
            judge=judge,
            sem=_Both(sem, global_sem),
        )
        async with lock:
            results.extend(rs)
            done += len(rs)
            with (out_dir / f"{spec.id.replace('/', '__')}.jsonl").open("a") as f:
                for r in rs:
                    f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            failed = [r for r in rs if not r.passed]
            mark = "ok " if not failed else "FAIL"
            detail = ""
            if failed:
                r = failed[0]
                detail = f" [{', '.join(r.kinds)}] raw={r.raw[:60]!r}" + (
                    f" err={r.error[:80]!r}" if r.error else ""
                )
            print(f"{done:5}/{total_steps} {mark} {spec.id:40} {case.name}{detail}", flush=True)

    # Case-major order interleaves providers, so the global pool is spread
    # across them instead of queueing on whichever provider comes first.
    jobs = [
        one(spec, case, a)
        for a in range(args.repeat)
        for case in cases
        for spec in specs
        if spec.tools or not case.tools
    ]
    total_steps = (
        sum(sum(t.infer for t in c.turns) for s in specs for c in cases if s.tools or not c.tools)
        * args.repeat
    )
    await asyncio.gather(*jobs)

    summary = write_summary(out_dir, results, args)
    print()
    print(summary)
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
