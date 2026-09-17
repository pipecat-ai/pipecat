#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A thin client for TypeSafe System One judgments.

TypeSafe's System One models (Jev) answer typed questions about a piece of
state instead of generating text: a ``Choice`` picks one of several labelled
options and reports a probability for each, a ``Noul`` returns the probability
that a yes/no statement holds, and a ``Score`` places the state on an ordered
rubric. Every question in a request is answered in parallel, so asking several
at once costs no extra latency.

:class:`TypeSafeJudge` wraps the async SDK client with the defaults a
real-time pipeline needs (a short timeout, no retries, one long-lived
connection) and converts responses into plain Pydantic models.
"""

import time
from collections.abc import Mapping
from typing import Any

from loguru import logger
from pydantic import BaseModel

try:
    from typesafe_sdk import (
        AsyncTypeSafeClient,
        Choice,
        ChoiceAnswer,
        Noul,
        NoulAnswer,
        NoulCriteria,
        RetryPolicy,
        Score,
        ScoreAnswer,
        SystemOneResponse,
        TypeSafeError,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use TypeSafe, you need to `uv add "pipecat-ai[typesafe]"`.')
    raise ImportError(f"Missing module: {e}") from e

__all__ = [
    "DEFAULT_RETRY",
    "Choice",
    "ChoiceDecision",
    "JudgeResult",
    "Noul",
    "NoulCriteria",
    "NoulDecision",
    "Question",
    "RetryPolicy",
    "Score",
    "ScoreDecision",
    "TypeSafeError",
    "TypeSafeJudge",
]

Question = Choice | Noul | Score
"""A question object from the SDK."""

DEFAULT_RETRY = RetryPolicy(
    max_retries=1,
    backoff_initial=0.1,
    backoff_max=0.2,
    api_timeout_error=False,
    timeout=None,
)
"""One retry after a short backoff for connection errors and retryable statuses; timeouts are not retried."""


class ChoiceDecision(BaseModel):
    """The answer to a ``Choice`` question.

    Parameters:
        choice: The selected option.
        confidence: How peaked the probability distribution is, from 0 to 1.
            It describes the choice, not whether acting on it is safe.
        probabilities: Probability of each option, keyed by option.
    """

    choice: str
    confidence: float
    probabilities: dict[str, float]


class NoulDecision(BaseModel):
    """The answer to a ``Noul`` (yes/no) question.

    Parameters:
        probability: Probability that the statement holds, from 0 to 1.
    """

    probability: float


class ScoreDecision(BaseModel):
    """The answer to a ``Score`` question.

    Parameters:
        score: Expected position on the rubric; may fall between two levels.
        confidence: How peaked the probability distribution is, from 0 to 1.
        probabilities: Probability of each rubric level, keyed by level.
    """

    score: float
    confidence: float
    probabilities: dict[int, float]


class JudgeResult(BaseModel):
    """All answers from one TypeSafe request, grouped by question type.

    Parameters:
        choices: ``Choice`` answers keyed by question id.
        nouls: ``Noul`` answers keyed by question id.
        scores: ``Score`` answers keyed by question id.
        model: The model that answered.
        latency_secs: Wall-clock time of the request, in seconds.
        input_tokens: Input tokens billed, when the API reports them.
        output_tokens: Output tokens billed, when the API reports them.
        request_id: The API request id, when the response carried one.
    """

    choices: dict[str, ChoiceDecision] = {}
    nouls: dict[str, NoulDecision] = {}
    scores: dict[str, ScoreDecision] = {}
    model: str
    latency_secs: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_id: str | None = None


class TypeSafeJudge:
    """Asks TypeSafe System One questions on behalf of pipeline components.

    One judge holds one HTTP connection. Create it once per bot and share it
    between the processors that need judgments; the component that created it
    closes it.

    The defaults suit a real-time pipeline: requests time out after one second,
    and a request that fails to connect or gets a retryable status (429, 5xx)
    is retried once after a short backoff. A request that times out is not
    retried, since a second wait would cost the turn more than the fallback
    every consumer has for a missing judgment.

    Example::

        judge = TypeSafeJudge()  # reads TYPESAFE_API_KEY
        judge.start()
        result = await judge.ask(
            {"bot_question": "Are you over 18?", "user_reply": "yeah I am"},
            {"answer": Choice(instructions="How did the user answer?",
                              criteria={"yes": "Confirms", "no": "Denies", "other": "Unclear"})},
        )
        result.choices["answer"].choice  # "yes"
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "jev-latest",
        timeout: float = 1.0,
        retry: RetryPolicy | None = None,
        base_url: str | None = None,
        client: AsyncTypeSafeClient | None = None,
    ):
        """Initialize the judge.

        Args:
            api_key: TypeSafe API key. Defaults to the ``TYPESAFE_API_KEY``
                environment variable.
            model: Model name sent with every request.
            timeout: Per-request HTTP timeout in seconds.
            retry: SDK retry policy. Defaults to :data:`DEFAULT_RETRY`: one
                retry after a 100 to 200 ms backoff for connection errors and
                retryable statuses, none for timeouts.
            base_url: API root override. Defaults to the ``TYPESAFE_BASE_URL``
                environment variable or the public API.
            client: An already-constructed SDK client to use instead of creating
                one. The judge never closes a client it did not create.
        """
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._retry = retry if retry is not None else DEFAULT_RETRY
        self._base_url = base_url
        self._client = client
        self._owns_client = client is None

    @property
    def model(self) -> str:
        """The model name sent with every request."""
        return self._model

    def start(self) -> None:
        """Create the SDK client if the judge was not given one.

        Safe to call more than once. Performs no I/O; call :meth:`warm_up`
        to open the connection ahead of the first judgment.
        """
        if self._client is None:
            self._client = AsyncTypeSafeClient(
                api_key=self._api_key,
                model=self._model,
                timeout=self._timeout,
                retry=self._retry,
                base_url=self._base_url,
            )

    async def warm_up(self) -> None:
        """Send one trivial request so the TLS connection is open before it matters.

        Failures are logged and swallowed: a cold connection is slower, not
        broken.
        """
        try:
            await self.ask("ok", {"warm_up": Noul(instructions="Is this the word ok?")})
        except Exception as e:
            logger.warning(f"TypeSafe warm-up failed: {e}")

    async def close(self) -> None:
        """Close the SDK client if the judge created it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def ask(self, state: str | Mapping[str, Any] | list, questions: Mapping[str, Question]):
        """Ask one or more questions about a piece of state.

        Args:
            state: The text or JSON the questions are about. Use an object with
                named fields when the context has several parts, and refer to
                them in instructions with backticked paths.
            questions: Questions keyed by an id of your choosing. Ids are for
                your code; the model never sees them.

        Returns:
            A :class:`JudgeResult` with every answer.

        Raises:
            TypeSafeError: The API rejected the request, timed out, or could not
                be reached.
        """
        if self._client is None:
            self.start()
        assert self._client is not None
        start = time.monotonic()
        response = await self._client.system_one(state, questions, timeout=self._timeout)
        return self._to_result(response, time.monotonic() - start)

    @staticmethod
    def _to_result(response: SystemOneResponse, latency_secs: float) -> JudgeResult:
        choices: dict[str, ChoiceDecision] = {}
        nouls: dict[str, NoulDecision] = {}
        scores: dict[str, ScoreDecision] = {}
        for question_id, answer in response.answers.items():
            if isinstance(answer, ChoiceAnswer):
                choices[question_id] = ChoiceDecision(
                    choice=answer.choice,
                    confidence=answer.confidence,
                    probabilities=dict(answer.probabilities),
                )
            elif isinstance(answer, NoulAnswer):
                nouls[question_id] = NoulDecision(probability=answer.noul)
            elif isinstance(answer, ScoreAnswer):
                scores[question_id] = ScoreDecision(
                    score=answer.score,
                    confidence=answer.confidence,
                    probabilities=dict(answer.probabilities),
                )
        try:
            request_id: str | None = response.request_id
        except TypeSafeError:
            request_id = None
        return JudgeResult(
            choices=choices,
            nouls=nouls,
            scores=scores,
            model=response.model,
            latency_secs=latency_secs,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            request_id=request_id,
        )
