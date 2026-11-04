#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""HTTP client for Hume's Audio Decisions API.

One :class:`HumeAudioDecisionsClient` holds one HTTP/2 connection pool, adds
the API key, retries when Hume is busy, and counts the input tokens every
request was billed. Several
:class:`~pipecat.classifiers.hume.audio_decisions.classifier.HumeAudioClassifier`
instances can share one.
"""

import asyncio
import json
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
    logger.error('In order to use Hume Audio Decisions, you need to `uv add "pipecat-ai[hume]"`.')
    raise ImportError(f"Missing module: {e}") from e

#: Statuses Hume answers when a retry can succeed: rate limited, a server
#: error, or at capacity.
_RETRY_STATUSES = (429, 500, 503)
#: The longest ``Retry-After`` the client waits out. An answer about the
#: user's turn is no use much later, so a longer wait is reported instead.
_MAX_RETRY_WAIT = 2.0
#: How long an idle connection is kept open, in seconds, so the next turn's
#: request doesn't pay for a new TLS handshake. Kept under the 60 seconds many
#: load balancers allow an idle connection.
_KEEPALIVE_EXPIRY = 50.0

#: Where the API is served.
DEFAULT_BASE_URL = "https://api.cloud.hume.ai"
#: Seconds to wait for a reply before giving up.
DEFAULT_TIMEOUT = 10.0


class HumeAudioDecisionsUsage(BaseModel):
    """Input tokens Hume billed: for one request, or over every request in :attr:`HumeAudioDecisionsClient.usage`.

    Parameters:
        input_tokens: Tokens billed for the audio, questions and context.
    """

    input_tokens: int = 0


class HumeAudioDecisionsClient:
    """HTTP client for Hume's ``/v0/decisions/file`` endpoint.

    Holds one HTTP/2 connection pool, kept open between questions. Retries
    with backoff when Hume answers 429, 500 or 503, waiting out its
    ``Retry-After`` when it gives one that's short enough. Counts the input
    tokens every request was billed in :attr:`usage`.

    :meth:`close` releases the connection pool.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 2,
    ):
        """Initialize the client.

        Args:
            api_key: Hume API key.
            base_url: Where the API is served.
            timeout: Seconds to wait for a reply before giving up.
            max_retries: How many times to retry a request Hume couldn't
                answer then.
        """
        if not api_key:
            raise ValueError("HumeAudioDecisionsClient needs an API key")
        self._max_retries = max_retries
        self._usage = HumeAudioDecisionsUsage()
        self._connected = False
        self._connect_lock = asyncio.Lock()
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"X-Hume-Api-Key": api_key},
            timeout=timeout,
            http2=True,
            limits=httpx.Limits(keepalive_expiry=_KEEPALIVE_EXPIRY),
        )

    @property
    def usage(self) -> HumeAudioDecisionsUsage:
        """Input tokens billed so far, over every request."""
        return self._usage

    async def connect(self):
        """Open the connection to Hume ahead of the first question.

        Lists one run, the lightest request the API serves, so the first
        question finds the connection open. It counts toward the rate limit
        like any request. Connects once: a client shared by several
        classifiers is asked by each of them, and only the first call sends
        anything.

        Raises:
            ClassifierError: If Hume could not be reached or refused the
                request.
        """
        async with self._connect_lock:
            if self._connected:
                return
            try:
                response = await self._http.get("/v0/decisions/run", params={"page_size": 1})
            except (httpx.HTTPError, OSError) as e:
                raise ClassifierError(f"could not connect to Hume: {e!r}") from e
            if response.status_code != 200:
                raise ClassifierError(self._rejection(response))
            self._connected = True

    async def close(self):
        """Close the connection pool."""
        self._connected = False
        await self._http.aclose()

    async def ask(
        self,
        wav: bytes,
        questions: Mapping[str, dict[str, Any]],
        context: str | None = None,
    ) -> tuple[dict[str, dict[str, Any]], HumeAudioDecisionsUsage]:
        """Send questions about one audio file and return Hume's answers.

        Args:
            wav: The audio, as a WAV file.
            questions: The questions by name, each in Hume's own format: a
                ``type`` of ``noul``, ``choice`` or ``score``,
                ``instructions``, and ``criteria``.
            context: A description of the recording, such as who is speaking
                and a transcript.

        Returns:
            The answers by the same names, in Hume's own format, and the
            input tokens the request was billed.

        Raises:
            ClassifierError: If Hume rejected the request, couldn't answer it
                after retrying, could not be reached, or left a question
                unanswered.
        """
        request: dict[str, Any] = {"questions": dict(questions)}
        if context:
            request["context"] = context
        # Hume reads the parts in this order: the request, then the file.
        files = [
            ("request", (None, json.dumps(request), "application/json")),
            ("file", ("audio.wav", wav, "audio/wav")),
        ]
        data = self._reply(await self._post(files))
        received = data.get("received")
        usage = HumeAudioDecisionsUsage(
            input_tokens=int(received.get("input_tokens", 0)) if isinstance(received, dict) else 0
        )
        self._usage.input_tokens += usage.input_tokens
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise ClassifierError("Hume reply has no answers")
        missing = [name for name in questions if not isinstance(answers.get(name), dict)]
        if missing:
            raise ClassifierError(f"Hume reply has no answer for {', '.join(missing)}")
        return {name: answers[name] for name in questions}, usage

    async def _post(self, files: list[tuple[str, tuple[Any, ...]]]) -> httpx.Response:
        """The reply to the request, retried while Hume can't answer it, or a ClassifierError."""
        response = await self._send(files)
        attempt = 0
        while response.status_code in _RETRY_STATUSES and attempt < self._max_retries:
            attempt += 1
            wait = self._retry_wait(response, attempt)
            logger.debug(f"Hume answered {response.status_code}, retrying in {wait}s")
            await asyncio.sleep(wait)
            response = await self._send(files)
        if response.status_code != 200:
            raise ClassifierError(self._rejection(response))
        return response

    async def _send(self, files: list[tuple[str, tuple[Any, ...]]]) -> httpx.Response:
        """One POST of the request, or a ClassifierError if Hume can't be reached."""
        try:
            return await self._http.post("/v0/decisions/file", files=files)
        except (httpx.HTTPError, OSError) as e:
            # Some httpx errors, such as a timeout, carry no message, and a
            # broken TLS connection can come through as an ssl.SSLError.
            raise ClassifierError(f"Hume request failed: {e!r}") from e

    def _retry_wait(self, response: httpx.Response, attempt: int) -> float:
        """Seconds to wait before retrying, or a ClassifierError when Hume asks for too long."""
        retry_after = response.headers.get("Retry-After")
        try:
            wait = float(retry_after) if retry_after else None
        except ValueError:
            wait = None
        if wait is None:
            wait = exponential_backoff_time(attempt, min_wait=0.25, max_wait=2.0, multiplier=0.25)
        if wait > _MAX_RETRY_WAIT:
            raise ClassifierError(
                f"Hume can't answer now (HTTP {response.status_code}), retry after {wait:g}s"
            )
        return wait

    def _reply(self, response: httpx.Response) -> dict[str, Any]:
        """The reply's JSON object, or a ClassifierError."""
        try:
            data = response.json()
        except ValueError as e:
            raise ClassifierError(f"Hume reply is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ClassifierError("Hume reply is not a JSON object")
        return data

    def _rejection(self, response: httpx.Response) -> str:
        """Why Hume rejected the request, from its error body when it has one."""
        message = f"Hume rejected the request: HTTP {response.status_code}"
        try:
            error = response.json()
        except ValueError:
            error = None
        if isinstance(error, dict):
            details = [str(error[key]) for key in ("code", "message", "param") if error.get(key)]
            if details:
                message += f": {' '.join(details)}"
        return message
