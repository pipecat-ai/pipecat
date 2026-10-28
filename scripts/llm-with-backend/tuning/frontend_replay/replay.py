#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Replay recorded user turns through a frontend model and count its delegate calls.

Each line of ``turns.jsonl`` is a user's turn from a run of the text eval
scenarios on one of the three example bots with a text frontend (OpenAI
Responses, Google and Anthropic): the frontend's context when the turn ended,
and whether the turn should be delegated. For each turn, the script builds the
frontend as the example does, with the example's instruction and the pairing
instruction and tool description as they are in the code, runs the model on
the context several times, and reports the turns where the model's decision
differs from the expected one.

One of the tuning checks: see ``scripts/llm-with-backend/tuning/README.md``.
Needs the frontend's key in ``.env``::

    uv run python scripts/llm-with-backend/tuning/frontend_replay/replay.py --frontend openai --samples 5
"""

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from loguru import logger

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "examples" / "multi-worker" / "llm-with-backend"))

from backend import FRONTEND_INSTRUCTIONS  # noqa: E402

from pipecat.frames.frames import (  # noqa: E402
    FunctionCallInProgressFrame,
    LLMContextFrame,
    LLMTextFrame,
)
from pipecat.pipeline.llm_with_backend import BackendConnector, ConnectorContext  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage  # noqa: E402
from pipecat.services.llm_service import LLMService  # noqa: E402
from pipecat.tests.utils import run_test  # noqa: E402


def _frontend(name: str) -> LLMService[Any]:
    """The frontend service of the example of that name."""
    if name == "openai":
        from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService

        return OpenAIResponsesLLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAIResponsesLLMService.Settings(system_instruction=FRONTEND_INSTRUCTIONS),
        )
    if name == "google":
        from pipecat.services.google.llm import GoogleLLMService

        return GoogleLLMService(
            api_key=os.environ["GOOGLE_API_KEY"],
            settings=GoogleLLMService.Settings(system_instruction=FRONTEND_INSTRUCTIONS),
        )
    if name == "anthropic":
        from pipecat.services.anthropic.llm import AnthropicLLMService

        return AnthropicLLMService(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            settings=AnthropicLLMService.Settings(system_instruction=FRONTEND_INSTRUCTIONS),
        )
    raise ValueError(f"unknown frontend {name!r}: expected openai, google or anthropic")


async def _run_once(frontend: str, messages: list[dict]) -> tuple[bool, str]:
    """Run the frontend on a context once: whether it called delegate, and what it said."""
    llm = _frontend(frontend)
    connector = BackendConnector()
    connector.bind(ConnectorContext(backend_name="backend", frontend_is_realtime=False))
    if connector.frontend_instruction:
        llm.append_system_instruction(connector.frontend_instruction)

    async def delegate(params):
        await params.result_callback({"status": "delegated", "backend": "working"})

    for tool in connector.tools:
        llm.register_function(tool.name, delegate, cancel_on_interruption=False)
        llm.get_llm_adapter().builtin_tools[tool.name] = tool
    # A service's own messages (a Google frontend's thought signatures) are
    # not replayed through another service.
    context = LLMContext(
        messages=cast(
            list[LLMContextMessage], [m for m in messages if m.get("role") != "llm_specific"]
        )
    )
    down, _ = await run_test(llm, frames_to_send=[LLMContextFrame(context)], start_timeout=5.0)
    called = any(isinstance(f, FunctionCallInProgressFrame) for f in down)
    text = "".join(f.text for f in down if isinstance(f, LLMTextFrame))
    return called, text


async def _replay(turns: list[dict], frontend: str, samples: int, concurrency: int) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)

    async def sample(index: int):
        async with semaphore:
            for attempt in range(3):
                try:
                    return index, await _run_once(frontend, turns[index]["messages"])
                except Exception as e:  # noqa: BLE001
                    error = e
                    await asyncio.sleep(2)
            logger.warning(f"turn {index}: {error}")
            return index, None

    runs: dict[int, list] = defaultdict(list)
    for coro in asyncio.as_completed(
        [sample(i) for i in range(len(turns)) for _ in range(samples)]
    ):
        index, result = await coro
        if result is not None:
            runs[index].append(result)
    return [{**turn, "runs": runs[i]} for i, turn in enumerate(turns)]


def _report(results: list[dict], show: bool):
    wrong_total = 0
    for result in results:
        runs = result["runs"]
        delegated = sum(1 for called, _ in runs if called)
        wrong = len(runs) - delegated if result["needs_delegate"] else delegated
        wrong_total += wrong
        user = [m for m in result["messages"] if m.get("role") == "user"][-1]["content"]
        expected = "delegate" if result["needs_delegate"] else "answer"
        mark = "!!" if wrong else "  "
        print(
            f"{mark} {expected:8} delegated {delegated:2}/{len(runs):<2}  "
            f"{result['frontend']}/{result['kind']}: {user[:60]}"
        )
        if show:
            for called, text in runs:
                print(f"       [{'D' if called else '-'}] {text[:150]!r}")
    print(f"\nwrong decisions: {wrong_total} over {len(results)} turns")


def main():
    parser = argparse.ArgumentParser(
        description="Replay recorded user turns through a frontend model and count its delegate calls."
    )
    parser.add_argument("--frontend", default="openai", help="openai, google or anthropic")
    parser.add_argument("--samples", type=int, default=5, help="Runs per turn.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--filter", help="Only turns whose user text contains this.")
    parser.add_argument("--show", action="store_true", help="Print what the model said each run.")
    parser.add_argument("--out", type=Path, help="Write each turn's runs to this JSON file.")
    args = parser.parse_args()
    load_dotenv(override=True)
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    turns = [json.loads(line) for line in (HERE / "turns.jsonl").open()]
    if args.filter:
        turns = [
            t
            for t in turns
            if args.filter in [m for m in t["messages"] if m.get("role") == "user"][-1]["content"]
        ]
    results = asyncio.run(_replay(turns, args.frontend, args.samples, args.concurrency))
    _report(results, args.show)
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
