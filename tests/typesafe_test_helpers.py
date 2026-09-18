#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A fake TypeSafe SDK client for tests that must not touch the network."""

import asyncio
from collections.abc import Mapping
from typing import Any

from typesafe_sdk import ChoiceAnswer, NoulAnswer, SystemOneResponse, Usage


def choice_response(
    choice: str,
    confidence: float,
    probabilities: dict[str, float] | None = None,
    *,
    question_id: str = "route",
    nouls: dict[str, float] | None = None,
) -> SystemOneResponse:
    """Build a response with one choice answer and optional noul answers."""
    answers: dict[str, Any] = {
        question_id: ChoiceAnswer(
            choice=choice,
            confidence=confidence,
            probabilities=probabilities or {choice: confidence},
        )
    }
    for noul_id, probability in (nouls or {}).items():
        answers[noul_id] = NoulAnswer(noul=probability)
    return SystemOneResponse(
        model="jev-test", usage=Usage(input_tokens=10, output_tokens=2), answers=answers
    )


class FakeTypeSafeClient:
    """Stands in for ``AsyncTypeSafeClient``.

    Returns ``response`` from every ``system_one`` call, after ``delay``
    seconds, or raises ``error`` instead. Records every request.
    """

    def __init__(
        self,
        response: SystemOneResponse | None = None,
        *,
        error: Exception | None = None,
        delay: float = 0.0,
    ):
        self.response = response
        self.error = error
        self.delay = delay
        self.requests: list[tuple[Any, Mapping[str, Any]]] = []
        self.closed = False

    async def system_one(self, state, questions, **kwargs) -> SystemOneResponse:
        self.requests.append((state, dict(questions)))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response

    async def aclose(self) -> None:
        self.closed = True
