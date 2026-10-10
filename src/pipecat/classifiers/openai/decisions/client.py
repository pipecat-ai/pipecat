#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""HTTP client for OpenAI's Decisions API.

One :class:`OpenAIDecisionsClient` holds one HTTP/2 connection pool, adds the
auth header, retries when OpenAI is busy, and counts the tokens every request
used. Several
:class:`~pipecat.classifiers.openai.decisions.classifier.OpenAIDecisionsClassifier`
instances can share one.
"""

import asyncio
from collections.abc import Mapping
from typing import Any

from loguru import logger
from pydantic import BaseModel

from pipecat.classifiers.base_classifier import ClassifierError
from pipecat.utils.network import exponential_backoff_time

try:
    import httpx
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use OpenAI Decisions, you need to `uv add "pipecat-ai[openai]"`.')
    raise ImportError(f"Missing module: {e}") from e

#: OpenAI answers a request with this status when the caller is rate limited.
_TOO_MANY_REQUESTS = 429
#: OpenAI answers a request with this status when it is temporarily overloaded.
_OVERLOADED = 503
#: How long an idle connection is kept open, in seconds. Questions come with
#: gaps between them, and a connection that has closed in the meantime costs a
#: new TLS handshake on the next one.
_KEEPALIVE_EXPIRY = 240.0

#: Where the API is served.
DEFAULT_BASE_URL = "https://api.openai.com/v1"
#: The decision model to ask.
DEFAULT_MODEL = "gpt-6-luna"
#: Seconds to wait for a reply before giving up.
DEFAULT_TIMEOUT = 10.0


class OpenAIDecisionsUsage(BaseModel):
    """Tokens the Decisions API used: for one request, or over every request.

    :attr:`OpenAIDecisionsClient.usage` holds the total over every request.

    Parameters:
        input_tokens: Tokens sent.
        output_tokens: Tokens received.
    """

    input_tokens: int = 0
    output_tokens: int = 0


class OpenAIDecisionsClient:
    """HTTP client for OpenAI's ``decisions`` endpoint.

    Holds one HTTP/2 connection pool, so many small requests share a
    connection and can be in flight at the same time. The connection is kept
    open between questions, so a gap between them does not cost a new TLS
    handshake. Retries with backoff when OpenAI answers 429 or 503. Counts the
    tokens every request used in :attr:`usage`. Errors carry the reason
    OpenAI gave.

    :meth:`connect` opens the connection ahead of the first question, and
    :meth:`close` releases it.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 3,
    ):
        """Initialize the client.

        Args:
            api_key: OpenAI API key.
            base_url: Where the API is served, such as
                ``https://eu.api.openai.com/v1`` for data residency in Europe.
            model: The decision model to ask.
            timeout: Seconds to wait for a reply before giving up.
            max_retries: How many times to retry a request OpenAI refused
                because it was busy.
        """
        if not api_key:
            raise ValueError("OpenAIDecisionsClient needs an API key")
        self._model = model
        self._max_retries = max_retries
        self._usage = OpenAIDecisionsUsage()
        self._connected = False
        self._connect_lock = asyncio.Lock()
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            http2=True,
            limits=httpx.Limits(keepalive_expiry=_KEEPALIVE_EXPIRY),
        )

    @property
    def model(self) -> str:
        """The decision model the questions go to."""
        return self._model

    @property
    def usage(self) -> OpenAIDecisionsUsage:
        """Tokens used so far, over every request."""
        return self._usage

    async def connect(self):
        """Open the connection to OpenAI.

        Looks up the model, a cheap request that also checks the key and
        whether it has access to the model. Connects once: a client shared by
        several classifiers is asked by each of them, and only the first call
        sends anything.

        Raises:
            ClassifierError: If OpenAI could not be reached or refused the
                request.
        """
        async with self._connect_lock:
            if self._connected:
                return
            try:
                response = await self._http.get(f"/models/{self._model}")
            except httpx.HTTPError as e:
                raise ClassifierError(f"could not connect to OpenAI Decisions: {e}") from e
            if response.status_code != 200:
                raise ClassifierError(
                    f"OpenAI Decisions refused the connection: {self._failure(response)}"
                )
            self._connected = True

    async def close(self):
        """Close the connection pool."""
        self._connected = False
        await self._http.aclose()

    async def ask(
        self, state: str | list[dict[str, Any]], questions: Mapping[str, dict[str, Any]]
    ) -> tuple[dict[str, dict[str, Any]], OpenAIDecisionsUsage]:
        """Send questions about one state and return OpenAI's answers.

        Args:
            state: What the questions are about, as the request's ``input``:
                text, or user messages.
            questions: The questions by name, each in OpenAI's own format: a
                ``type`` of ``predicate``, ``choice`` or ``score``,
                ``instructions``, and ``choices`` or ``levels``.

        Returns:
            The answers by the same names, in OpenAI's own format, and the
            tokens the request used.

        Raises:
            ClassifierError: If OpenAI rejected the request, kept refusing it
                because it was busy, could not be reached, or left a question
                unanswered.
        """
        body = {
            "model": self._model,
            "input": state,
            "questions": [{"name": name, **question} for name, question in questions.items()],
        }
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._http.post("/decisions", json=body)
            except httpx.HTTPError as e:
                # Some httpx errors, such as a timeout, carry no message.
                raise ClassifierError(f"OpenAI Decisions request failed: {e!r}") from e
            if response.status_code in (_TOO_MANY_REQUESTS, _OVERLOADED):
                if attempt > self._max_retries:
                    raise ClassifierError(
                        f"OpenAI Decisions is busy ({self._failure(response)}) "
                        f"after {attempt} attempts"
                    )
                wait = exponential_backoff_time(
                    attempt, min_wait=0.25, max_wait=2.0, multiplier=0.25
                )
                logger.debug(
                    f"OpenAI Decisions answered {response.status_code}, retrying in {wait}s"
                )
                await asyncio.sleep(wait)
                continue
            if response.status_code != 200:
                raise ClassifierError(
                    f"OpenAI Decisions rejected the request: {self._failure(response)}"
                )
            try:
                data = response.json()
            except ValueError as e:
                raise ClassifierError(f"OpenAI Decisions reply is not valid JSON: {e}") from e
            if not isinstance(data, dict):
                raise ClassifierError("OpenAI Decisions reply is not a JSON object")
            usage = data.get("usage")
            request_usage = (
                OpenAIDecisionsUsage.model_validate(usage)
                if isinstance(usage, dict)
                else OpenAIDecisionsUsage()
            )
            self._usage.input_tokens += request_usage.input_tokens
            self._usage.output_tokens += request_usage.output_tokens
            answers = data.get("answers")
            if not isinstance(answers, list):
                raise ClassifierError("OpenAI Decisions reply has no answers")
            # Answers come back as a list, each carrying its question's name.
            by_name = {
                answer["name"]: answer
                for answer in answers
                if isinstance(answer, dict) and isinstance(answer.get("name"), str)
            }
            missing = [name for name in questions if name not in by_name]
            if missing:
                raise ClassifierError(
                    f"OpenAI Decisions reply has no answer for {', '.join(missing)}"
                )
            return {name: by_name[name] for name in questions}, request_usage

    def _failure(self, response: httpx.Response) -> str:
        """A failed response's HTTP status and the reason OpenAI gave."""
        status = f"HTTP {response.status_code}"
        try:
            message = response.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            return status
        return f"{status}: {message}" if message else status
