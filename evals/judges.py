#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The classifiers the eval suites judge with, named by ``judge.eval.factory:``.

A factory takes the ``judge.eval:`` block and returns the classifier that
decides the verdicts. The suites run from the repository root (``run.sh``
goes there), so a block names one as ``evals.judges.<name>``.
"""

import os

from pipecat.classifiers.base_classifier import BaseClassifier
from pipecat.classifiers.cloudflare.clef.classifier import ClefClassifier
from pipecat.classifiers.cloudflare.clef.client import DEFAULT_MODEL as CLEF_DEFAULT_MODEL
from pipecat.classifiers.openai.decisions.classifier import OpenAIDecisionsClassifier
from pipecat.classifiers.openai.decisions.client import (
    DEFAULT_BASE_URL as OPENAI_DEFAULT_BASE_URL,
)
from pipecat.classifiers.openai.decisions.client import (
    DEFAULT_MODEL as OPENAI_DEFAULT_MODEL,
)
from pipecat.classifiers.typesafe.jev.classifier import JevClassifier
from pipecat.classifiers.typesafe.jev.client import DEFAULT_BASE_URL, DEFAULT_MODEL

# Seconds to wait for Jev, Clef or OpenAI's Decisions API to answer a question.
# All three answer in a few hundred milliseconds, so a question still waiting
# this long is one to ask again.
TIMEOUT = 2.5


def typesafe_classifier(config: dict) -> BaseClassifier:
    """TypeSafe's Jev, a hosted classifier that answers in a few hundred milliseconds.

    Args:
        config: The ``judge.eval:`` block, with an optional ``model`` and
            ``endpoint``; the client's defaults otherwise. The API key comes
            from ``TYPESAFE_API_KEY``.

    Returns:
        The classifier.

    Raises:
        ValueError: If there is no ``TYPESAFE_API_KEY``.
    """
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise ValueError("Judging with Jev needs an API key: set TYPESAFE_API_KEY.")
    return JevClassifier(
        api_key=api_key,
        model=config.get("model") or DEFAULT_MODEL,
        base_url=str(config.get("endpoint") or DEFAULT_BASE_URL).rstrip("/"),
        timeout=TIMEOUT,
    )


def cloudflare_classifier(config: dict) -> BaseClassifier:
    """Cloudflare's Clef, a classifier on Workers AI that answers in a few hundred milliseconds.

    Args:
        config: The ``judge.eval:`` block, with an optional ``model``
            (``clef`` or ``clef-flash``). The account comes from
            ``CLOUDFLARE_ACCOUNT_ID`` and the API token from
            ``CLOUDFLARE_API_KEY``.

    Returns:
        The classifier.

    Raises:
        ValueError: If ``CLOUDFLARE_ACCOUNT_ID`` or ``CLOUDFLARE_API_KEY`` is
            not set.
    """
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_key = os.environ.get("CLOUDFLARE_API_KEY")
    if not account_id or not api_key:
        raise ValueError(
            "Judging with Clef needs an account and a token: "
            "set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_KEY."
        )
    return ClefClassifier(
        account_id=account_id,
        api_key=api_key,
        model=config.get("model") or CLEF_DEFAULT_MODEL,
        timeout=TIMEOUT,
    )


def openai_classifier(config: dict) -> BaseClassifier:
    """OpenAI's Decisions API, a classifier that answers in a few hundred milliseconds.

    Args:
        config: The ``judge.eval:`` block, with an optional ``model`` and
            ``endpoint``; the client's defaults otherwise. The API key comes
            from ``OPENAI_API_KEY``.

    Returns:
        The classifier.

    Raises:
        ValueError: If there is no ``OPENAI_API_KEY``.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Judging with OpenAI Decisions needs an API key: set OPENAI_API_KEY.")
    return OpenAIDecisionsClassifier(
        api_key=api_key,
        model=config.get("model") or OPENAI_DEFAULT_MODEL,
        base_url=str(config.get("endpoint") or OPENAI_DEFAULT_BASE_URL).rstrip("/"),
        timeout=TIMEOUT,
    )
