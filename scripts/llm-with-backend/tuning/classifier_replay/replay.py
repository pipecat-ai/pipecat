#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Replay recorded user turns through predictive delegation's classifier.

Each line of ``turns.jsonl`` is a user's turn from a conversation with one of
the text example bots: the frontend's context when the turn ended, and whether
the turn needs the backend. For each turn, the script builds the classifier's
input the way ``PredictiveDelegation`` does in a running bot, from the
example's frontend instruction and the pairing instruction as they are in the
code, asks ``JevClassifier``, and reports the turns it gets wrong at the
threshold for delegating early.

One of the tuning checks: see ``scripts/llm-with-backend/tuning/README.md``.
Needs ``TYPESAFE_API_KEY``::

    uv run python scripts/llm-with-backend/tuning/classifier_replay/replay.py --out after.json
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "examples" / "multi-worker" / "llm-with-backend"))

from backend import FRONTEND_INSTRUCTIONS  # noqa: E402

from pipecat.classifiers.base_classifier import YesNoQuestion  # noqa: E402
from pipecat.classifiers.typesafe.jev.classifier import JevClassifier  # noqa: E402
from pipecat.frames.frames import LLMContextFrame  # noqa: E402
from pipecat.pipeline.llm_with_backend import LLMWithBackend, PredictiveDelegation  # noqa: E402
from pipecat.pipeline.llm_with_backend.predictive_delegation import (  # noqa: E402
    _DELEGATION_PREDICTION_THRESHOLD,
)
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService  # noqa: E402


def _bind(classifier: JevClassifier) -> PredictiveDelegation:
    """A PredictiveDelegation bound as in the OpenAI Responses example.

    Binding composes the frontend's system instruction and builds the
    classifier's question, so the replay asks what a running bot would.
    """
    predictive_delegation = PredictiveDelegation(classifier)
    LLMWithBackend(
        frontend=OpenAIResponsesLLMService(
            api_key="unused",
            settings=OpenAIResponsesLLMService.Settings(system_instruction=FRONTEND_INSTRUCTIONS),
        ),
        backend="backend",
        predictive_delegation=predictive_delegation,
    )
    return predictive_delegation


async def _replay(turns: list[dict]) -> list[dict]:
    classifier = JevClassifier(api_key=os.environ["TYPESAFE_API_KEY"])
    predictive_delegation = _bind(classifier)
    question = predictive_delegation._question
    assert question is not None
    await classifier.client.connect()
    try:
        # A first call warms the connection up and isn't timed.
        await classifier.yes_no("hello", {"q": YesNoQuestion(instructions="Is this a greeting?")})
        results = []
        for turn in turns:
            frame = LLMContextFrame(LLMContext(turn["messages"]))
            state = predictive_delegation._classifier_state(frame)
            started = time.perf_counter()
            answers = await classifier.yes_no(state, {"needs_backend": question})
            results.append(
                {
                    "utterance": turn["messages"][-1]["content"],
                    "frontend": turn["frontend"],
                    "run": turn["run"],
                    "needs_backend": turn["needs_backend"],
                    "probability": answers["needs_backend"].probability,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                }
            )
    finally:
        await classifier.client.close()
    return results


def _report(results: list[dict]) -> None:
    threshold = _DELEGATION_PREDICTION_THRESHOLD
    yes = [r for r in results if r["needs_backend"]]
    no = [r for r in results if not r["needs_backend"]]
    false_yes = sum(r["probability"] >= threshold for r in no)
    missed_yes = sum(r["probability"] < threshold for r in yes)
    latencies = sorted(r["latency_ms"] for r in results)
    print(f"{len(results)} turns, threshold {threshold}")
    print(f'  false "yes" (delegated early, not needed): {false_yes}/{len(no)}')
    print(f'  missed "yes" (left to the frontend):       {missed_yes}/{len(yes)}')
    print(
        f"  latency: p50 {statistics.median(latencies):.0f}ms, "
        f"p90 {latencies[int(len(latencies) * 0.9) - 1]}ms"
    )
    print('\nprobability of a "yes" per utterance: median (min-max)')
    for utterance in dict.fromkeys(r["utterance"] for r in results):
        rows = [r for r in results if r["utterance"] == utterance]
        p = sorted(r["probability"] for r in rows)
        label = "yes" if rows[0]["needs_backend"] else "no"
        print(f"  {label:3} {statistics.median(p):.2f} ({p[0]:.2f}-{p[-1]:.2f})  {utterance}")


def main():
    parser = argparse.ArgumentParser(
        description="Replay recorded user turns through predictive delegation's classifier."
    )
    parser.add_argument("--out", type=Path, help="Write each turn's result to this JSON file.")
    args = parser.parse_args()
    load_dotenv(override=True)
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    turns = [json.loads(line) for line in (HERE / "turns.jsonl").open()]
    results = asyncio.run(_replay(turns))
    _report(results)
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
