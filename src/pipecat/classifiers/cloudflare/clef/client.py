#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""HTTP client for Clef, Cloudflare's decision model on Workers AI.

One :class:`ClefClient` holds one HTTP/2 connection pool, adds the auth
header, retries when Workers AI is busy, and counts the tokens every request
used. Several :class:`~pipecat.classifiers.cloudflare.clef.classifier.ClefClassifier`
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
    logger.error('In order to use Clef, you need to `uv add "pipecat-ai[cloudflare]"`.')
    raise ImportError(f"Missing module: {e}") from e

#: Workers AI answers a request with this status when it is out of capacity.
_TOO_MANY_REQUESTS = 429
#: How long an idle connection is kept open, in seconds. Questions come with
#: gaps between them, and a connection that has closed in the meantime costs a
#: new TLS handshake on the next one.
_KEEPALIVE_EXPIRY = 240.0

#: Where Cloudflare's API is served.
CLOUDFLARE_API_URL = "https://api.cloudflare.com/client/v4"
#: The Clef model to ask.
DEFAULT_MODEL = "clef"
#: Seconds to wait for a reply before giving up.
DEFAULT_TIMEOUT = 10.0


class ClefUsage(BaseModel):
    """Tokens Clef used: for one request, or over every request in :attr:`ClefClient.usage`.

    Parameters:
        input_tokens: Tokens sent.
        output_tokens: Tokens received.
    """

    input_tokens: int = 0
    output_tokens: int = 0


class ClefClient:
    """HTTP client for Clef on Workers AI.

    Holds one HTTP/2 connection pool, so many small requests share a
    connection and can be in flight at the same time. The connection is kept
    open between questions, so a gap between them does not cost a new TLS
    handshake. Retries with backoff when Workers AI answers 429. Counts the
    tokens every request used in :attr:`usage`. Errors carry the reasons
    Cloudflare gave.

    :meth:`connect` opens the connection ahead of the first question, and
    :meth:`close` releases it.
    """

    def __init__(
        self,
        *,
        account_id: str,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 3,
    ):
        """Initialize the client.

        Args:
            account_id: Cloudflare account ID.
            api_key: Cloudflare API token with Workers AI access.
            model: The Clef model to ask: ``clef``, or ``clef-flash`` for
                faster answers.
            timeout: Seconds to wait for a reply before giving up.
            max_retries: How many times to retry a request Workers AI
                refused because it was busy.
        """
        if not account_id:
            raise ValueError("ClefClient needs a Cloudflare account ID")
        if not api_key:
            raise ValueError("ClefClient needs a Cloudflare API token")
        self._model = model
        self._max_retries = max_retries
        self._usage = ClefUsage()
        self._connected = False
        self._connect_lock = asyncio.Lock()
        self._http = httpx.AsyncClient(
            base_url=f"{CLOUDFLARE_API_URL}/accounts/{account_id}/ai",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            http2=True,
            limits=httpx.Limits(keepalive_expiry=_KEEPALIVE_EXPIRY),
        )

    @property
    def model(self) -> str:
        """The Clef model the questions go to."""
        return self._model

    @property
    def usage(self) -> ClefUsage:
        """Tokens used so far, over every request."""
        return self._usage

    async def connect(self):
        """Open the connection to Workers AI.

        Asks for the model's schema, a cheap request that also checks the
        token and the model name. Connects once: a client shared by several
        classifiers is asked by each of them, and only the first call sends
        anything.

        Raises:
            ClassifierError: If Workers AI could not be reached or refused
                the request.
        """
        async with self._connect_lock:
            if self._connected:
                return
            try:
                response = await self._http.get(
                    "/models/schema", params={"model": f"@cf/cloudflare/{self._model}"}
                )
            except httpx.HTTPError as e:
                raise ClassifierError(f"could not connect to Clef: {e}") from e
            if response.status_code != 200:
                raise ClassifierError(f"Clef refused the connection: {self._failure(response)}")
            self._connected = True

    async def close(self):
        """Close the connection pool."""
        self._connected = False
        await self._http.aclose()

    async def ask(
        self, state: str | dict[str, Any] | list[Any], questions: Mapping[str, dict[str, Any]]
    ) -> tuple[dict[str, dict[str, Any]], ClefUsage]:
        """Send questions about one state and return Clef's answers.

        Args:
            state: What the questions are about.
            questions: The questions by name, each in Clef's own format: a
                ``type`` of ``noul``, ``choice`` or ``score``,
                ``instructions``, and ``criteria``.

        Returns:
            The answers by the same names, in Clef's own format, and the
            tokens the request used.

        Raises:
            ClassifierError: If Clef rejected the request, kept refusing it
                because it was busy, could not be reached, or left a question
                unanswered.
        """
        body = {"model": self._model, "state": state, "questions": dict(questions)}
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._http.post(f"/run/@cf/cloudflare/{self._model}", json=body)
            except httpx.HTTPError as e:
                # Some httpx errors, such as a timeout, carry no message.
                raise ClassifierError(f"Clef request failed: {e!r}") from e
            if response.status_code == _TOO_MANY_REQUESTS:
                if attempt > self._max_retries:
                    raise ClassifierError(
                        f"Clef is busy ({self._failure(response)}) after {attempt} attempts"
                    )
                wait = exponential_backoff_time(
                    attempt, min_wait=0.25, max_wait=2.0, multiplier=0.25
                )
                logger.debug(f"Clef answered {response.status_code}, retrying in {wait}s")
                await asyncio.sleep(wait)
                continue
            if response.status_code != 200:
                raise ClassifierError(f"Clef rejected the request: {self._failure(response)}")
            result = self._result(response)
            usage = result.get("usage")
            request_usage = (
                ClefUsage.model_validate(usage) if isinstance(usage, dict) else ClefUsage()
            )
            self._usage.input_tokens += request_usage.input_tokens
            self._usage.output_tokens += request_usage.output_tokens
            answers = result.get("answers")
            if not isinstance(answers, dict):
                raise ClassifierError("Clef reply has no answers")
            missing = [name for name in questions if not isinstance(answers.get(name), dict)]
            if missing:
                raise ClassifierError(f"Clef reply has no answer for {', '.join(missing)}")
            return {name: answers[name] for name in questions}, request_usage

    def _result(self, response: httpx.Response) -> dict[str, Any]:
        """The reply inside Cloudflare's ``result`` envelope."""
        try:
            data = response.json()
        except ValueError as e:
            raise ClassifierError(f"Clef reply is not valid JSON: {e}") from e
        result = data.get("result") if isinstance(data, dict) else None
        if not isinstance(result, dict):
            raise ClassifierError("Clef reply has no result")
        return result

    def _failure(self, response: httpx.Response) -> str:
        """A failed response's HTTP status and the reasons Cloudflare gave."""
        status = f"HTTP {response.status_code}"
        try:
            errors = response.json().get("errors")
        except (ValueError, AttributeError):
            return status
        reasons = [
            str(error["message"])
            for error in errors or []
            if isinstance(error, dict) and error.get("message")
        ]
        return f"{status}: {'; '.join(reasons)}" if reasons else status
