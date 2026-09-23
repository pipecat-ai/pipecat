#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Ask a classifier the three kinds of question about a call transcript.

A classifier is not a pipeline component: build one, ask it, and clean it up.
By default the questions go to Jev; with ``--llm`` the same questions go to an
OpenAI model through ``LLMClassifier`` instead.

Usage::

    TYPESAFE_API_KEY=... python features-classifiers.py
    OPENAI_API_KEY=... python features-classifiers.py --llm gpt-4o-mini
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv

from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceQuestion,
    ChoiceResult,
    ScoreQuestion,
    ScoreResult,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.classifiers.jev.classifier import JevClassifier
from pipecat.classifiers.llm.classifier import LLMClassifier
from pipecat.metrics.metrics import LLMUsageMetricsData, ProcessingMetricsData
from pipecat.services.openai.llm import OpenAILLMService

load_dotenv(override=True)

# The state is what the questions are about. Text works; structured data such
# as a transcript with speaker labels tells the classifier more.
TRANSCRIPT = [
    {"role": "assistant", "content": "Thanks for calling Acme. How can I help you today?"},
    {
        "role": "user",
        "content": "I was charged twice for March. This is the third time I'm calling.",
    },
    {"role": "assistant", "content": "I'm sorry about that. Let me look at the account."},
    {"role": "user", "content": "Honestly, if this isn't fixed today I'm cancelling."},
]

QUESTIONS = {
    "refund": YesNoQuestion(
        instructions="Does the customer want money back?",
        yes="asks for a refund, a credit, or a charge to be reversed",
        no="anything else",
    ),
    "department": ChoiceQuestion(
        instructions="Which team should take this call?",
        options={
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, errors",
            "sales": "pricing, new contracts",
            "other": None,
        },
    ),
    "mood": ScoreQuestion(
        instructions="How upset is the customer?",
        levels=["calm", "frustrated", "angry"],
    ),
}


async def main(classifier: BaseClassifier):
    # After every call the classifier reports how long it took and, when it
    # knows, the tokens it used. A processor would push these as a MetricsFrame.
    @classifier.event_handler("on_metrics")
    async def on_metrics(classifier, data):
        for item in data:
            if isinstance(item, ProcessingMetricsData):
                print(f"  took {item.value * 1000:.0f} ms")
            elif isinstance(item, LLMUsageMetricsData):
                print(f"  used {item.value.prompt_tokens} + {item.value.completion_tokens} tokens")

    try:
        # All three questions about one state in one call. ask() returns a
        # result of each question's kind; the typed methods below return one kind.
        results = await classifier.ask(TRANSCRIPT, QUESTIONS)

        refund = results["refund"]
        assert isinstance(refund, YesNoResult)
        print(f"refund: {'yes' if refund.is_yes else 'no'} ({refund.probability:.2f})")

        department = results["department"]
        assert isinstance(department, ChoiceResult)
        print(f"department: {department.choice} ({department.confidence:.2f})")
        for option, probability in department.probabilities.items():
            print(f"  {option}: {probability:.2f}")

        mood = results["mood"]
        assert isinstance(mood, ScoreResult)
        print(f"mood: {mood.score:.2f} on a scale of {len(mood.levels)} ({mood.confidence:.2f})")
        for level in mood.levels:
            print(f"  {level.level}: {level.probability:.2f}")

        # A typed method takes questions of one kind and returns typed results.
        results = await classifier.yes_no(
            TRANSCRIPT, {"churn": YesNoQuestion(instructions="Might the customer leave?")}
        )
        print(f"churn: {results['churn'].probability:.2f}")
    finally:
        await classifier.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", metavar="MODEL", help="use an OpenAI model instead of Jev")
    args = parser.parse_args()

    if args.llm:
        llm = OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMService.Settings(model=args.llm),
        )
        asyncio.run(main(LLMClassifier(llm=llm)))
    else:
        asyncio.run(main(JevClassifier(api_key=os.environ["TYPESAFE_API_KEY"])))
