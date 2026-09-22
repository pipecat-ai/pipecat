#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""HTTP client for Jev, TypeSafe's hosted classification model.

One :class:`JevClient` holds one HTTP/2 connection pool, adds the auth
header, retries when Jev is busy, and counts the tokens every request used.
Several :class:`~pipecat.classifiers.jev.JevClassifier` instances can share
one.
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
    logger.error("In order to use Jev, you need to `pip install pipecat-ai[jev]`.")
    raise Exception(f"Missing module: {e}")

#: Jev answers a request with this status when the caller is rate limited.
_TOO_MANY_REQUESTS = 429
#: Jev answers a request with this status when it is temporarily overloaded.
_OVERLOADED = 529
#: How long an idle connection is kept open, in seconds. Questions come with
#: gaps between them, and a connection that has closed in the meantime costs a
#: new TLS handshake on the next one. Jev closes idle connections after about
#: five minutes, so this stays under that.
_KEEPALIVE_EXPIRY = 240.0


class JevUsage(BaseModel):
    """Tokens a :class:`JevClient` has used so far.

    Parameters:
        input_tokens: Tokens sent, over every request.
        output_tokens: Tokens received, over every request.
    """

    input_tokens: int = 0
    output_tokens: int = 0


class JevClient:
    """HTTP client for Jev's ``systemone`` endpoint.

    Holds one HTTP/2 connection pool, so many small requests share a
    connection and can be in flight at the same time. The connection is kept
    open between questions, so a gap between them does not cost a new TLS
    handshake. Retries with backoff when Jev answers 429 or 529. Counts the
    tokens every request used in :attr:`usage`.

    :meth:`connect` opens the connection ahead of the first question, and
    :meth:`close` releases it.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.typesafe.ai",
        model: str = "jev-latest",
        timeout: float = 10.0,
        max_retries: int = 3,
        http_client: httpx.AsyncClient | None = None,
    ):
        """Initialize the client.

        Args:
            api_key: Jev API key.
            base_url: Where the API is served.
            model: The Jev model to ask.
            timeout: Seconds to wait for a reply before giving up.
            max_retries: How many times to retry a request Jev refused
                because it was busy.
            http_client: An HTTP client to send requests with instead of
                the one built here. Mostly for tests.
        """
        if not api_key:
            raise ValueError("JevClient needs an API key")
        self._model = model
        self._max_retries = max_retries
        self._usage = JevUsage()
        self._connected = False
        self._connect_lock = asyncio.Lock()
        self._http = http_client or httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            http2=True,
            limits=httpx.Limits(keepalive_expiry=_KEEPALIVE_EXPIRY),
        )
        if http_client is not None:
            self._http.base_url = httpx.URL(base_url)
            self._http.headers["Authorization"] = f"Bearer {api_key}"

    @property
    def usage(self) -> JevUsage:
        """Tokens used so far, over every request."""
        return self._usage

    async def connect(self):
        """Open the connection to Jev.

        Lists the models, the cheapest request there is, so the first
        question finds the connection already open. Connects once: a client
        shared by several classifiers is asked by each of them, and only the
        first call sends anything.

        Raises:
            ClassifierError: If Jev could not be reached or refused the
                request.
        """
        async with self._connect_lock:
            if self._connected:
                return
            try:
                response = await self._http.get("/v1/models")
            except httpx.HTTPError as e:
                raise ClassifierError(f"could not connect to Jev: {e}") from e
            if response.status_code != 200:
                raise ClassifierError(f"Jev refused the connection: HTTP {response.status_code}")
            self._connected = True

    async def close(self):
        """Close the connection pool."""
        await self._http.aclose()

    async def ask(
        self, state: str | dict[str, Any] | list[Any], questions: Mapping[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Send questions about one state and return Jev's answers.

        Args:
            state: What the questions are about.
            questions: The questions by name, each in Jev's own format: a
                ``type`` of ``noul``, ``choice`` or ``score``,
                ``instructions``, and ``criteria``.

        Returns:
            The answers by the same names, in Jev's own format.

        Raises:
            ClassifierError: If Jev rejected the request, kept refusing it
                because it was busy, could not be reached, or left a question
                unanswered.
        """
        body = {"model": self._model, "state": state, "questions": dict(questions)}
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._http.post("/v1/systemone", json=body)
            except httpx.HTTPError as e:
                raise ClassifierError(f"Jev request failed: {e}") from e
            if response.status_code in (_TOO_MANY_REQUESTS, _OVERLOADED):
                if attempt > self._max_retries:
                    raise ClassifierError(
                        f"Jev is busy (HTTP {response.status_code}) after {attempt} attempts"
                    )
                wait = exponential_backoff_time(
                    attempt, min_wait=0.25, max_wait=2.0, multiplier=0.25
                )
                logger.debug(f"Jev answered {response.status_code}, retrying in {wait}s")
                await asyncio.sleep(wait)
                continue
            if response.status_code != 200:
                raise ClassifierError(f"Jev rejected the request: HTTP {response.status_code}")
            data = response.json()
            usage = data.get("usage") or {}
            self._usage.input_tokens += usage.get("input_tokens", 0)
            self._usage.output_tokens += usage.get("output_tokens", 0)
            answers = data.get("answers") if isinstance(data, dict) else None
            if not isinstance(answers, dict):
                raise ClassifierError("Jev reply has no answers")
            missing = [name for name in questions if not isinstance(answers.get(name), dict)]
            if missing:
                raise ClassifierError(f"Jev reply has no answer for {', '.join(missing)}")
            return {name: answers[name] for name in questions}
