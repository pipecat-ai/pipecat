#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Release evals orchestrator.

For every (example, scenario) pair in ``TESTS`` this launches the example bot
with the eval transport on its own port, then drives it with the eval harness
(the harness connects as an RTVI client, synthesizes the user turns, transcribes
the bot's speech, and judges it). Several run concurrently.

No Daily room and no second "eval bot" agent: the harness replaces both.

Usage::

    uv run run-release-evals.py                  # run everything
    uv run run-release-evals.py -p voice-openai  # only matching examples
    uv run run-release-evals.py -c 8 -a          # 8 at a time, record audio

Results stream live as each eval finishes (out of order, since they run
concurrently). Per-example bot logs (and optional recordings) land under
test-runs/<name>/ — check those for detail rather than a verbose flag, which
would be unreadable interleaved across concurrent runs.
"""

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from pipecat.evals.harness import EvalResult, run_scenario
from pipecat.evals.scenario import describe_config, load_scenario

load_dotenv(override=True)

_console = Console()

SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = SCRIPT_DIR.parent.parent / "examples"
SCENARIOS_DIR = SCRIPT_DIR / "scenarios"

# Each example bot binds its own port so they can run in parallel.
BASE_PORT = 7900
# How long to wait for a freshly launched bot to start listening.
BOT_READY_TIMEOUT_SECS = 60.0
# How long to wait for a bot subprocess to exit after we ask it to stop.
BOT_STOP_TIMEOUT_SECS = 10.0

# ---------------------------------------------------------------------------
# Tests: which scenario each example bot runs.
#
# Scenarios are files in scenarios/<name>.yaml and are reusable, so one shared
# scenario covers many examples. Grouped by category for readability; add an
# entry per example.
# ---------------------------------------------------------------------------

SIMPLE_MATH = "simple_math"
WEATHER_FUNCTION_CALL = "weather_function_call"
WEATHER_AND_RESTAURANT = "weather_and_restaurant"
GREETING = "greeting"
MULTI_TURN = "multi_turn"
INTERRUPTION = "interruption"
LANGUAGE_SWITCH = "language_switch"

# Voice: the same simple-math conversation over audio for every service.
# Excluded (need infra we don't drive here, or image input): voice-google-image
# (vision), voice-xtts (local XTTS docker), voice-*-sagemaker (AWS SageMaker).
TESTS_VOICE = [
    ("voice/voice-aicoustics.py", SIMPLE_MATH),
    ("voice/voice-assemblyai.py", SIMPLE_MATH),
    ("voice/voice-assemblyai-turn-detection.py", SIMPLE_MATH),
    ("voice/voice-asyncai.py", SIMPLE_MATH),
    ("voice/voice-asyncai-http.py", SIMPLE_MATH),
    ("voice/voice-aws.py", SIMPLE_MATH),
    ("voice/voice-aws-strands.py", WEATHER_FUNCTION_CALL),  # a weather agent, not math
    ("voice/voice-azure.py", SIMPLE_MATH),
    ("voice/voice-azure-http.py", SIMPLE_MATH),
    ("voice/voice-camb.py", SIMPLE_MATH),
    ("voice/voice-cartesia.py", SIMPLE_MATH),
    ("voice/voice-cartesia-http.py", SIMPLE_MATH),
    ("voice/voice-cartesia-turns.py", SIMPLE_MATH),
    ("voice/voice-deepgram.py", SIMPLE_MATH),
    ("voice/voice-deepgram-flux.py", SIMPLE_MATH),
    ("voice/voice-deepgram-http.py", SIMPLE_MATH),
    ("voice/voice-elevenlabs.py", SIMPLE_MATH),
    ("voice/voice-elevenlabs-http.py", SIMPLE_MATH),
    ("voice/voice-fal.py", SIMPLE_MATH),
    ("voice/voice-fish.py", SIMPLE_MATH),
    ("voice/voice-gladia.py", SIMPLE_MATH),
    ("voice/voice-gladia-vad.py", SIMPLE_MATH),
    ("voice/voice-google.py", SIMPLE_MATH),
    ("voice/voice-google-audio-in.py", SIMPLE_MATH),
    ("voice/voice-google-gemini-tts.py", SIMPLE_MATH),
    ("voice/voice-google-http.py", SIMPLE_MATH),
    ("voice/voice-gradium.py", SIMPLE_MATH),
    ("voice/voice-groq.py", SIMPLE_MATH),
    ("voice/voice-hume.py", SIMPLE_MATH),
    ("voice/voice-inworld.py", SIMPLE_MATH),
    ("voice/voice-inworld-http.py", SIMPLE_MATH),
    ("voice/voice-kokoro.py", SIMPLE_MATH),
    ("voice/voice-krisp-viva.py", SIMPLE_MATH),
    ("voice/voice-langchain.py", SIMPLE_MATH),
    ("voice/voice-lmnt.py", SIMPLE_MATH),
    ("voice/voice-minimax.py", SIMPLE_MATH),
    ("voice/voice-mistral.py", SIMPLE_MATH),
    ("voice/voice-neuphonic.py", SIMPLE_MATH),
    ("voice/voice-neuphonic-http.py", SIMPLE_MATH),
    ("voice/voice-nvidia.py", SIMPLE_MATH),
    ("voice/voice-openai.py", SIMPLE_MATH),
    ("voice/voice-openai-http.py", SIMPLE_MATH),
    ("voice/voice-openai-responses.py", SIMPLE_MATH),
    ("voice/voice-openai-responses-http.py", SIMPLE_MATH),
    ("voice/voice-piper.py", SIMPLE_MATH),
    ("voice/voice-resemble.py", SIMPLE_MATH),
    ("voice/voice-rime.py", SIMPLE_MATH),
    ("voice/voice-rime-http.py", SIMPLE_MATH),
    ("voice/voice-sarvam.py", SIMPLE_MATH),
    ("voice/voice-sarvam-http.py", SIMPLE_MATH),
    ("voice/voice-smallest.py", SIMPLE_MATH),
    ("voice/voice-soniox.py", SIMPLE_MATH),
    ("voice/voice-speechmatics.py", SIMPLE_MATH),
    ("voice/voice-speechmatics-vad.py", SIMPLE_MATH),
    ("voice/voice-xai.py", SIMPLE_MATH),
    ("voice/voice-xai-http.py", SIMPLE_MATH),
]

# Function calling: a weather tool call over audio. A few examples also run the
# two-tool weather_and_restaurant scenario. Excluded: *-video (image input ->
# vision), function-calling-direct / -missing-handler (behavior demos, not
# service checks), -ollama (local model), -together (known broken).
TESTS_FUNCTION_CALLING = [
    ("getting-started/07-function-calling.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-anthropic.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-anthropic-async.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-anthropic-async-stream.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-aws.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-azure.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-cerebras.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-deepseek.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-fireworks.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-google.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-google-async.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-google-async-stream.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-google-vertex.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-grok.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-groq.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-inception.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-mistral.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-nebius.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-novita.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-nvidia.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-openai.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-openai-async.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-openai-async-stream.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-openai-responses.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-openai-responses-async.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-openai-responses-async-stream.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-openai-responses-http.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-openrouter.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-perplexity.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-qwen.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-sambanova.py", WEATHER_FUNCTION_CALL),
    ("function-calling/function-calling-sarvam.py", WEATHER_FUNCTION_CALL),
    # Same examples, two-tool scenario.
    ("getting-started/07-function-calling.py", WEATHER_AND_RESTAURANT),
    ("function-calling/function-calling-anthropic.py", WEATHER_AND_RESTAURANT),
    ("function-calling/function-calling-aws.py", WEATHER_AND_RESTAURANT),
    ("function-calling/function-calling-google.py", WEATHER_AND_RESTAURANT),
    ("function-calling/function-calling-openai-responses.py", WEATHER_AND_RESTAURANT),
    ("function-calling/function-calling-openai-responses-http.py", WEATHER_AND_RESTAURANT),
]

# Behavioral scenarios exercise pipeline features (greeting, context, barge-in)
# rather than a specific service, so they run against a couple representative
# bots rather than every example. Mostly text modality (fast, deterministic);
# audio coverage already comes from the per-service scenarios above.
TESTS_BEHAVIOR = [
    ("voice/voice-cartesia.py", GREETING),
    ("voice/voice-cartesia.py", MULTI_TURN),
    ("voice/voice-cartesia.py", INTERRUPTION),
    ("voice/voice-openai.py", INTERRUPTION),
    ("voice/voice-cartesia.py", LANGUAGE_SWITCH),
]

TESTS: list[tuple[str, str]] = [
    *TESTS_VOICE,
    *TESTS_FUNCTION_CALLING,
    *TESTS_BEHAVIOR,
]


@dataclass
class TestState:
    """Mutable per-eval state, updated in place so the live display can read it."""

    example: str
    scenario: str
    status: str = "pending"  # pending | running | done
    result: EvalResult | None = None
    error: str | None = None
    started_at: float | None = None
    duration_ms: int | None = None


def _verdict(s: TestState) -> str:
    """Collapse a state into a display verdict."""
    if s.status != "done":
        return s.status  # pending | running
    if s.error or s.result is None:
        return "error"
    if s.result.skipped:
        return "skipped"
    return "passed" if s.result.passed else "failed"


async def _wait_for_port(host: str, port: int, timeout: float) -> bool:
    """Return True once a TCP connection to host:port succeeds (server is up)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            _, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return True
        except OSError:
            await asyncio.sleep(0.25)
    return False


async def _stop_bot(proc: asyncio.subprocess.Process) -> None:
    """Ask the bot subprocess to stop, escalating to kill if it lingers."""
    if proc.returncode is not None:
        return
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=BOT_STOP_TIMEOUT_SECS)
    except TimeoutError:
        proc.kill()
        await proc.wait()


async def run_one(
    state: TestState,
    port: int,
    logs_dir: Path,
    record_dir: Path | None,
    cache_dir: str | None,
    sem: asyncio.Semaphore,
) -> None:
    """Launch one example bot, run its scenario, and record the outcome on ``state``."""
    async with sem:
        # Include the scenario in the filename: the same example can run several
        # scenarios concurrently, and they must not share a log/recording file.
        safe = f"{state.example.replace('/', '_')}__{state.scenario}"
        example_path = EXAMPLES_DIR / state.example
        scenario_path = SCENARIOS_DIR / f"{state.scenario}.yaml"
        log_path = logs_dir / f"{safe}.log"

        state.status = "running"
        state.started_at = time.monotonic()
        proc: asyncio.subprocess.Process | None = None
        logf = None
        try:
            if not example_path.exists():
                state.error = f"example not found: {example_path}"
                return
            if not scenario_path.exists():
                state.error = f"scenario not found: {scenario_path}"
                return

            logf = log_path.open("wb")
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(example_path),
                "-t",
                "eval",
                "--port",
                str(port),
                stdout=logf,
                stderr=asyncio.subprocess.STDOUT,
            )

            if not await _wait_for_port("localhost", port, BOT_READY_TIMEOUT_SECS):
                state.error = f"bot did not start (see {log_path})"
                return

            scenario = load_scenario(scenario_path)
            record_path = str(record_dir / f"{safe}.wav") if record_dir else None
            state.result = await run_scenario(
                scenario, f"ws://localhost:{port}", record_path=record_path, cache_dir=cache_dir
            )
        except Exception as e:
            state.error = f"error: {e}"
        finally:
            if proc is not None:
                await _stop_bot(proc)
            if logf is not None:
                logf.close()
            state.status = "done"
            if state.result is not None:
                state.duration_ms = state.result.duration_ms
            elif state.started_at is not None:
                state.duration_ms = int((time.monotonic() - state.started_at) * 1000)
            if not _console.is_terminal:
                _print_line(state)


# --- display -----------------------------------------------------------------

_VERDICT_GLYPH = {
    "passed": ("✓", "green", "32"),
    "failed": ("✗", "red", "31"),
    "skipped": ("⊘", "yellow", "33"),
    "error": ("✗", "red", "31"),
}


def _status_cell(s: TestState):
    """A rich renderable for the status column (spinner while running)."""
    if s.status == "pending":
        return Text("·", style="dim")
    if s.status == "running":
        return Spinner("dots", style="cyan")
    glyph, style, _ = _VERDICT_GLYPH[_verdict(s)]
    return Text(glyph, style=style)


class _Dashboard:
    """Live table rendered every frame from the shared list of states."""

    def __init__(self, states: list[TestState]):
        self.states = states

    def __rich__(self) -> Group:
        table = Table.grid(padding=(0, 2))
        table.add_column()  # status
        table.add_column()  # example
        table.add_column()  # scenario
        table.add_column(justify="right")  # timing
        for s in self.states:
            if s.status == "running" and s.started_at is not None:
                detail = f"{int(time.monotonic() - s.started_at)}s"
            elif s.status == "done" and s.duration_ms is not None:
                detail = f"{s.duration_ms}ms"
            else:
                detail = ""
            table.add_row(
                _status_cell(s),
                Text(s.example),
                Text(s.scenario, style="cyan"),
                Text(detail, style="dim"),
            )
        done = sum(1 for s in self.states if s.status == "done")
        passed = sum(1 for s in self.states if _verdict(s) == "passed")
        summary = Text(
            f"{passed}/{len(self.states)} passed  ·  {done}/{len(self.states)} done", "bold"
        )
        return Group(table, Text(""), summary)


def _c(text: str, code: str) -> str:
    """ANSI-colour text when stdout is a TTY (used for the non-live fallback)."""
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


def _print_line(s: TestState) -> None:
    """Print a single result line (non-TTY fallback, streamed as each finishes)."""
    glyph, _, code = _VERDICT_GLYPH[_verdict(s)]
    extra = f"({s.duration_ms}ms)" if s.duration_ms is not None and not s.error else (s.error or "")
    print(f"  {_c(glyph, code)} {s.example} {_c(s.scenario, '36')} {_c(extra, '2')}", flush=True)


def _finalize(states: list[TestState], runs_dir: Path) -> int:
    """Print failure detail and the final tally; return the process exit code."""
    failed = [s for s in states if _verdict(s) in ("failed", "error")]
    passed = sum(1 for s in states if _verdict(s) == "passed")
    skipped = sum(1 for s in states if _verdict(s) == "skipped")

    if failed:
        print()
        for s in failed:
            if s.error:
                print(
                    f"  {_c('✗', '31')} {s.example} {_c(s.scenario, '36')} {_c('— ' + s.error, '2')}"
                )
            elif s.result is not None:
                print(f"  {_c('✗', '31')} {s.example} {_c(s.scenario, '36')}")
                for f in s.result.failures:
                    print(f"      {_c('•', '31')} {f}")

    print()
    print(f"  {passed}/{len(states)} passed" + (f", {skipped} skipped" if skipped else ""))
    print(f"  logs: {runs_dir}")
    print()
    return 0 if not failed else 1


async def main(args: argparse.Namespace) -> int:
    # Keep the console clean for the live display; per-example logs go to files.
    logger.remove()

    pairs = TESTS
    if args.pattern:
        pairs = [(e, s) for (e, s) in pairs if args.pattern in e]
    if not pairs:
        print("No examples match.")
        return 1

    name = args.name or datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = SCRIPT_DIR / "test-runs" / name
    logs_dir = runs_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    record_dir = (runs_dir / "recordings") if args.audio else None
    if record_dir:
        record_dir.mkdir(parents=True, exist_ok=True)

    # Print each distinct scenario's config up front (per-eval would interleave
    # under concurrency).
    print("Scenarios:")
    seen: dict[str, str] = {}
    for _, scenario_name in pairs:
        if scenario_name not in seen:
            try:
                seen[scenario_name] = describe_config(
                    load_scenario(SCENARIOS_DIR / f"{scenario_name}.yaml")
                )
            except Exception as e:
                seen[scenario_name] = f"(failed to load: {e})"
    for scenario_name, cfg in seen.items():
        print(f"  {scenario_name}: {cfg}")
    print()

    states = [TestState(example=e, scenario=s) for (e, s) in pairs]
    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        run_one(states[i], BASE_PORT + i, logs_dir, record_dir, args.cache_dir, sem)
        for i in range(len(states))
    ]

    if _console.is_terminal:
        # Live dashboard: each task mutates its state; the table re-renders every
        # frame, so rows flip pending -> running -> result as they finish.
        with Live(_Dashboard(states), console=_console, refresh_per_second=12.5):
            await asyncio.gather(*tasks)
    else:
        # Piped output: stream a line per eval as it completes (run_one prints).
        print(f"Running {len(states)} eval(s), {args.concurrency} at a time...")
        await asyncio.gather(*tasks)

    return _finalize(states, runs_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat release evals (harness-driven)")
    parser.add_argument("-p", "--pattern", help="Only run examples whose path contains this")
    parser.add_argument("-c", "--concurrency", type=int, default=4, help="How many to run at once")
    parser.add_argument("-a", "--audio", action="store_true", help="Record conversation audio")
    parser.add_argument("-n", "--name", help="Run name (defaults to a timestamp)")
    parser.add_argument("--cache-dir", help="Directory for cached synthesized user audio")
    args = parser.parse_args()

    sys.exit(asyncio.run(main(args)))
