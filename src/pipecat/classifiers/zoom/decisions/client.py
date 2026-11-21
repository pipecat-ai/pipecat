#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""HTTP client for the Zoom Decisions API.

One :class:`ZoomDecisionsClient` holds one HTTP/2 connection pool, adds the
auth and tracking headers, retries when Zoom is busy, and counts the tokens
every request used. Several
:class:`~pipecat.classifiers.zoom.decisions.classifier.ZoomDecisionsClassifier`
instances can share one.
"""

import asyncio
import time
from typing import Any

from loguru import logger
from pydantic import BaseModel

from pipecat.classifiers.base_classifier import ClassifierError
from pipecat.services.zoom_ai.utils import check_url, new_tracking_id, request_headers
from pipecat.utils.network import exponential_backoff_time

try:
    import httpx
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Zoom Decisions, you need to `uv add "pipecat-ai[zoom-ai]"`.')
    raise ImportError(f"Missing module: {e}") from e

#: Zoom answers a request with this status when the caller is rate limited.
_TOO_MANY_REQUESTS = 429
#: Zoom answers a request with this status when it is temporarily overloaded.
_OVERLOADED = 503
#: How long an idle connection is kept open, in seconds. Questions come with
#: gaps between them, and a connection that has closed in the meantime costs a
#: new TLS handshake on the next one.
_KEEPALIVE_EXPIRY = 240.0

#: Where questions are sent.
DEFAULT_URL = "https://api.zoom.us/v2/aiservices/decisions/decide"
#: The decision model to ask: ``zoom-decisions`` or ``zoom-decisions-pro``.
DEFAULT_MODEL = "zoom-decisions"
#: Seconds to wait for an answer, retries included, before giving up.
DEFAULT_TIMEOUT = 10.0


class ZoomDecisionsUsage(BaseModel):
    """Tokens the Decisions API used: for one request, or over every request.

    :attr:`ZoomDecisionsClient.usage` holds the total over every request.

    Parameters:
        input_tokens: Tokens sent, as Zoom counts them in ``input_units``.
        output_tokens: Tokens received, one per question.
        cached_input_tokens: Tokens sent that Zoom read from its cache.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0


class ZoomDecisionsClient:
    """HTTP client for the Zoom Decisions ``decide`` endpoint.

    Holds one HTTP/2 connection pool, so many small requests share a
    connection and can be in flight at the same time. The connection is kept
    open between questions, so a gap between them does not cost a new TLS
    handshake. Retries with backoff when Zoom answers 429 or 503, waiting at
    least as long as Zoom's ``Retry-After`` asks, within one time bound for the
    whole request. Counts the tokens every request used in :attr:`usage`.
    Errors carry the reason Zoom gave and the request's ``x-zm-trackingid``,
    which Zoom support uses to find it.

    :meth:`connect` opens the connection ahead of the first question, and
    :meth:`close` releases it.
    """

    def __init__(
        self,
        *,
        api_key: str,
        url: str = DEFAULT_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 3,
    ):
        """Initialize the client.

        Args:
            api_key: Zoom AI Services API key.
            url: The ``decide`` endpoint. A plaintext ``http://`` URL to a
                remote host logs a warning.
            model: The decision model to ask: ``zoom-decisions`` or
                ``zoom-decisions-pro``.
            timeout: Seconds to wait for an answer, retries included, before
                giving up.
            max_retries: How many times to retry a request Zoom refused
                because it was busy.
        """
        if not api_key:
            raise ValueError("ZoomDecisionsClient needs an API key")
        self._api_key = api_key
        self._url = check_url(url, service="Zoom Decisions", data="questions")
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._usage = ZoomDecisionsUsage()
        self._connected = False
        self._connect_lock = asyncio.Lock()
        self._http = httpx.AsyncClient(
            timeout=timeout,
            http2=True,
            limits=httpx.Limits(keepalive_expiry=_KEEPALIVE_EXPIRY),
        )

    @property
    def model(self) -> str:
        """The decision model the questions go to."""
        return self._model

    @property
    def usage(self) -> ZoomDecisionsUsage:
        """Tokens used so far, over every request."""
        return self._usage

    async def connect(self):
        """Open the connection to Zoom.

        Sends an ``OPTIONS`` request to the endpoint, which opens the
        connection without asking anything. Zoom has no request that checks
        the key without asking a question, so a bad key shows on the first
        question. Connects once: a client shared by several classifiers is
        asked by each of them, and only the first call sends anything.

        Raises:
            ClassifierError: If Zoom could not be reached.
        """
        async with self._connect_lock:
            if self._connected:
                return
            try:
                await self._http.options(self._url)
            except (httpx.HTTPError, OSError) as e:
                # A broken TLS connection can come through as an ssl.SSLError.
                raise ClassifierError(f"could not connect to Zoom Decisions: {e!r}") from e
            self._connected = True

    async def close(self):
        """Close the connection pool."""
        self._connected = False
        await self._http.aclose()

    async def ask(
        self, state: str | list[dict[str, Any]], questions: list[dict[str, Any]], **options: Any
    ) -> tuple[dict[str, dict[str, Any]], ZoomDecisionsUsage]:
        """Send questions about one state and return Zoom's answers.

        Args:
            state: What the questions are about, as the request's ``input``:
                text, or user messages.
            questions: The questions in Zoom's own format, each with a
                distinct ``name``.
            **options: Other request fields, such as
                ``enable_image_input=True``.

        Returns:
            The answers by question name, in Zoom's own format, and the tokens
            the request used.

        Raises:
            ClassifierError: If Zoom rejected the request, kept refusing it
                because it was busy, could not be reached, did not answer in
                time, or left a question unanswered.
        """
        body = {"model": self._model, "input": state, "questions": questions, **options}
        try:
            data = await asyncio.wait_for(self._post(body), self._timeout)
        except TimeoutError as e:
            raise ClassifierError(f"Zoom Decisions did not answer within {self._timeout:g}s") from e
        usage = data.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        details = usage.get("details")
        details = details if isinstance(details, dict) else {}
        request_usage = ZoomDecisionsUsage(
            input_tokens=_count(usage.get("input_units")),
            output_tokens=_count(usage.get("output_units")),
            cached_input_tokens=_count(details.get("cached_input_tokens")),
        )
        self._usage.input_tokens += request_usage.input_tokens
        self._usage.output_tokens += request_usage.output_tokens
        self._usage.cached_input_tokens += request_usage.cached_input_tokens
        answers = data.get("answers")
        if not isinstance(answers, list):
            raise ClassifierError("Zoom Decisions reply has no answers")
        # Answers come back as a list, each carrying its question's name.
        by_name = {
            answer["name"]: answer
            for answer in answers
            if isinstance(answer, dict) and isinstance(answer.get("name"), str)
        }
        names = [question["name"] for question in questions]
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ClassifierError(f"Zoom Decisions reply has no answer for {', '.join(missing)}")
        return {name: by_name[name] for name in names}, request_usage

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """Send the request, retrying while Zoom is busy, and return the reply."""
        deadline = time.monotonic() + self._timeout
        attempt = 0
        while True:
            attempt += 1
            tracking_id = new_tracking_id()
            try:
                response = await self._http.post(
                    self._url, json=body, headers=request_headers(self._api_key, tracking_id)
                )
            except (httpx.HTTPError, OSError) as e:
                # Some httpx errors, such as a timeout, carry no message, and a
                # broken TLS connection can come through as an ssl.SSLError.
                raise ClassifierError(
                    f"Zoom Decisions request failed: {e!r} (tracking_id={tracking_id})"
                ) from e
            if response.status_code in (_TOO_MANY_REQUESTS, _OVERLOADED):
                wait = max(
                    exponential_backoff_time(attempt, min_wait=0.25, max_wait=2.0, multiplier=0.25),
                    _retry_after(response),
                )
                if attempt > self._max_retries or time.monotonic() + wait > deadline:
                    raise ClassifierError(
                        f"Zoom Decisions is busy ({self._failure(response, tracking_id)}) "
                        f"after {attempt} attempts"
                    )
                logger.debug(f"Zoom Decisions answered {response.status_code}, retrying in {wait}s")
                await asyncio.sleep(wait)
                continue
            if response.status_code != 200:
                raise ClassifierError(
                    f"Zoom Decisions rejected the request: {self._failure(response, tracking_id)}"
                )
            try:
                data = response.json()
            except ValueError as e:
                raise ClassifierError(f"Zoom Decisions reply is not valid JSON: {e}") from e
            if not isinstance(data, dict):
                raise ClassifierError("Zoom Decisions reply is not a JSON object")
            return data

    def _failure(self, response: httpx.Response, tracking_id: str) -> str:
        """A failed response's HTTP status, the reason Zoom gave, and the tracking ID."""
        status = f"HTTP {response.status_code}"
        try:
            message = response.json().get("message")
        except (ValueError, AttributeError):
            message = None
        reason = f"{status}: {message}" if message else status
        return f"{reason} (tracking_id={tracking_id})"


def _retry_after(response: httpx.Response) -> float:
    """The seconds Zoom's ``Retry-After`` header asks to wait, or 0."""
    try:
        return max(float(response.headers.get("Retry-After", 0)), 0.0)
    except ValueError:
        return 0.0


def _count(value: Any) -> int:
    """A token count from the reply, or 0 if there is none."""
    return value if isinstance(value, int) else 0
