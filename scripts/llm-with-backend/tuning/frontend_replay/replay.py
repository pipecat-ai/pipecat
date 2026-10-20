#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Replay recorded user turns through a frontend model, counting its delegate calls and recording what it says around them.

Each line of ``turns.jsonl`` is a user's turn from a run of the text eval
scenarios on one of the three example bots with a text frontend (OpenAI
Responses, Google and Anthropic): the frontend's context when the turn ended,
and whether the turn should be delegated. For each turn, the script builds the
frontend as the example does, with the example's instruction and the pairing
instruction and tool description as they are in the code, runs the model on
the context several times, and reports the turns where the model's decision
differs from the expected one. A turn the model delegates runs on, as in the
bot: the ``delegate`` call settles and the model replies, so the acknowledgement
the user would hear, what the model said with the call and in that reply, is
reported too, with the words it took and how often it was said twice (with
the call and again after it).

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

from pipecat.frames.frames import LLMContextFrame  # noqa: E402
from pipecat.pipeline.llm_with_backend import (  # noqa: E402
    DELEGATE_TOOL_NAME,
    BackendConnector,
    ConnectorContext,
)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage  # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (  # noqa: E402
    LLMContextAggregatorPair,
)
from pipecat.services.llm_service import LLMService  # noqa: E402
from pipecat.tests.utils import SleepFrame, run_test  # noqa: E402

#: How long a turn is given for the model's reply after a delegate call.
REPLY_SECS = 8


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


async def _run_once(frontend: str, messages: list[dict]) -> tuple[bool, list[str]]:
    """Run the frontend on a context once: whether it called delegate, and what it said.

    The frontend runs ahead of an assistant aggregator, as in the bot, so a
    delegate call settles and the model replies to its result. What it said is
    read back from the context: the text it wrote with the call, if any, then
    the reply.
    """
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
    recorded = len(context.get_messages())
    aggregator = LLMContextAggregatorPair(context).assistant()
    await run_test(
        Pipeline([llm, aggregator]),
        frames_to_send=[LLMContextFrame(context), SleepFrame(sleep=REPLY_SECS)],
        start_timeout=5.0,
    )
    added = [m for m in context.get_messages()[recorded:] if isinstance(m, dict)]
    called = any(
        call.get("function", {}).get("name") == DELEGATE_TOOL_NAME
        for m in added
        if m.get("role") == "assistant"
        for call in m.get("tool_calls") or []
    )
    said = [
        content
        for m in added
        if m.get("role") == "assistant" and isinstance(content := m.get("content"), str) and content
    ]
    return called, said


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


def _words(said: list[str]) -> int:
    return sum(len(text.split()) for text in said)


def _report(results: list[dict], show: bool):
    wrong_total = 0
    acks: list[list[str]] = []
    for result in results:
        runs = result["runs"]
        delegated = sum(1 for called, _ in runs if called)
        wrong = len(runs) - delegated if result["needs_delegate"] else delegated
        wrong_total += wrong
        acks.extend(said for called, said in runs if called)
        user = [m for m in result["messages"] if m.get("role") == "user"][-1]["content"]
        expected = "delegate" if result["needs_delegate"] else "answer"
        mark = "!!" if wrong else "  "
        print(
            f"{mark} {expected:8} delegated {delegated:2}/{len(runs):<2}  "
            f"{result['frontend']}/{result['kind']}: {user[:60]}"
        )
        if show:
            for called, said in runs:
                print(f"       [{'D' if called else '-'}] {' | '.join(said)[:150]!r}")
    print(f"\nwrong decisions: {wrong_total} over {len(results)} turns")
    if acks:
        words = sorted(_words(said) for said in acks)
        twice = sum(1 for said in acks if len(said) > 1)
        print(
            f"acknowledgements: {len(acks)}, words median {words[len(words) // 2]}, "
            f"max {words[-1]}, said twice {twice}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Replay recorded user turns through a frontend model, counting its delegate calls and recording what it says around them."
    )
    parser.add_argument("--frontend", default="openai", help="openai, google or anthropic")
    parser.add_argument("--samples", type=int, default=5, help="Runs per turn.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--filter", help="Only turns whose user text contains this.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print what the model said each run, with the call and after it.",
    )
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
