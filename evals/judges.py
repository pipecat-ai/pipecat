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
from pipecat.classifiers.jev.classifier import JevClassifier
from pipecat.classifiers.jev.client import DEFAULT_BASE_URL, DEFAULT_MODEL

# Seconds to wait for Jev to answer a question. Jev answers in a few hundred
# milliseconds, so a question still waiting this long is one to ask again.
JEV_TIMEOUT = 2.5


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
        timeout=JEV_TIMEOUT,
    )
