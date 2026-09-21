#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Measure a classifier on the turn-completion question.

Replays labeled user turns through a classifier, one at a time as the turn
path would, and reports how often it agrees with the label and how long each
answer takes. The labeled turns come from ``turn_completion.yaml`` next to
this script, plus every user line of the scripted release scenarios, which
are complete turns by construction.

Usage::

    TYPESAFE_API_KEY=... python evals/classifiers/measure_turn_completion.py
    python evals/classifiers/measure_turn_completion.py --no-context
    python evals/classifiers/measure_turn_completion.py --repeat 3
    OPENAI_API_KEY=... python evals/classifiers/measure_turn_completion.py --llm gpt-4o-mini
"""

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

import yaml

from pipecat.classifiers.base_classifier import BaseClassifier, ClassifierError
from pipecat.classifiers.jev.classifier import JevClassifier
from pipecat.classifiers.llm.classifier import LLMClassifier
from pipecat.turns.user_stop.classifier_user_turn_completion_stop_strategy import (
    TURN_COMPLETION_QUESTION,
)

HERE = Path(__file__).parent
SCENARIOS = HERE.parent / "release" / "scenarios" / "scripted"


def load_turns(with_context: bool) -> list[dict]:
    turns = yaml.safe_load(open(HERE / "turn_completion.yaml"))["turns"]

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("!include", lambda loader, node: None)
    for path in sorted(SCENARIOS.glob("*.yaml")):
        # The turn-completion scenarios cut lines off on purpose; the labeled
        # set above covers those.
        if path.name.startswith(("filter_incomplete_turns", "classifier_turn_completion")):
            continue
        data = yaml.load(open(path), Loader=Loader) or {}
        for scenario in data.get("scenarios") or []:
            for turn in scenario.get("turns") or []:
                user = turn.get("user")
                if isinstance(user, str) and user.strip():
                    turns.append({"user": _as_stt(user), "label": "complete", "source": path.name})
    if not with_context:
        for turn in turns:
            turn.pop("bot", None)
    return turns


def _as_stt(text: str) -> str:
    """Make a scripted line look like a transcript: lower case, no punctuation."""
    return "".join(c for c in text.lower() if c.isalnum() or c in " '").strip()


async def measure(classifier: BaseClassifier, turns: list[dict], repeat: int) -> None:
    rows = []
    for _ in range(repeat):
        for turn in turns:
            state = {"user": turn["user"]}
            if turn.get("bot"):
                state = {"assistant": turn["bot"], "user": turn["user"]}
            started = time.perf_counter()
            try:
                result = (await classifier.choice(state, {"turn": TURN_COMPLETION_QUESTION}))[
                    "turn"
                ]
                predicted, confidence = result.choice, result.confidence
            except ClassifierError as e:
                predicted, confidence = f"error: {e}", 0.0
            rows.append(
                {
                    **turn,
                    "predicted": predicted,
                    "confidence": confidence,
                    "ms": (time.perf_counter() - started) * 1000,
                }
            )
    report(rows)


def report(rows: list[dict]) -> None:
    labels = ["complete", "short", "long"]
    latencies = sorted(r["ms"] for r in rows)
    agree = sum(r["predicted"] == r["label"] for r in rows)
    print(f"\n{len(rows)} turns")
    print(
        f"latency  p50 {statistics.median(latencies):.0f} ms   "
        f"p95 {latencies[int(len(latencies) * 0.95) - 1]:.0f} ms   "
        f"max {latencies[-1]:.0f} ms"
    )
    print(f"agreement {agree}/{len(rows)} = {agree / len(rows):.1%}")
    incomplete = [r for r in rows if r["label"] != "complete"]
    false_complete = [r for r in incomplete if r["predicted"] == "complete"]
    print(
        f"false complete {len(false_complete)}/{len(incomplete)} = "
        f"{len(false_complete) / max(len(incomplete), 1):.1%}  (incomplete turns judged complete)"
    )
    print("\nconfusion (rows: label, columns: predicted)")
    print(f"{'':10s}" + "".join(f"{p:>10s}" for p in labels + ["error"]))
    for label in labels:
        counts = []
        for predicted in labels:
            counts.append(sum(r["label"] == label and r["predicted"] == predicted for r in rows))
        counts.append(sum(r["label"] == label and r["predicted"].startswith("error") for r in rows))
        print(f"{label:10s}" + "".join(f"{c:>10d}" for c in counts))
    misses = [r for r in rows if r["predicted"] != r["label"]]
    if misses:
        print("\ndisagreements")
        for r in misses:
            bot = f"  (bot: {r['bot']})" if r.get("bot") else ""
            print(
                f"  {r['label']:8s} -> {r['predicted']:8s} conf={r['confidence']:.2f}  "
                f"{r['user']!r}{bot}"
            )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-context", action="store_true", help="drop the bot's previous line")
    parser.add_argument("--repeat", type=int, default=1, help="run the set this many times")
    parser.add_argument("--llm", metavar="MODEL", help="use an OpenAI model instead of Jev")
    args = parser.parse_args()

    turns = load_turns(with_context=not args.no_context)
    print(f"{len(turns)} labeled turns, context {'off' if args.no_context else 'on'}")

    if args.llm:
        await measure_with_llm(args.llm, turns, args.repeat)
        return

    api_key = os.getenv("TYPESAFE_API_KEY")
    if not api_key:
        sys.exit("TYPESAFE_API_KEY is not set")
    classifier = JevClassifier(api_key=api_key)
    try:
        await measure(classifier, turns, args.repeat)
        usage = classifier.client.usage
        print(f"\ntokens  in {usage.input_tokens}  out {usage.output_tokens}")
    finally:
        await classifier.cleanup()


async def measure_with_llm(model: str, turns: list[dict], repeat: int) -> None:
    """Run the measurement with an LLM classifier."""
    from pipecat.services.openai.llm import OpenAILLMService

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY is not set")
    classifier = LLMClassifier(llm=OpenAILLMService(api_key=api_key, model=model))
    try:
        await measure(classifier, turns, repeat)
    finally:
        await classifier.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
