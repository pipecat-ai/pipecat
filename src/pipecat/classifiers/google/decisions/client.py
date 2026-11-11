#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Client for Google's Decisions API.

One :class:`GoogleDecisionsClient` holds one ``google-genai`` client on an
HTTP/2 connection pool, bounds every request in time, and counts the tokens
every request used. Several
:class:`~pipecat.classifiers.google.decisions.classifier.GoogleDecisionsClassifier`
instances can share one.
"""

import asyncio
from collections.abc import Mapping
from typing import Any, cast

from loguru import logger
from pydantic import BaseModel

from pipecat.classifiers.base_classifier import ClassifierError

try:
    import google.genai as genai
    import httpx
    from google.genai import types
    from google.genai.decisions import Answer, InputParam, QuestionParam
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Google Decisions, you need to `uv add "pipecat-ai[google]"`.')
    raise ImportError(f"Missing module: {e}") from e

#: How long an idle connection is kept open, in seconds. Questions come with
#: gaps between them, and a connection that has closed in the meantime costs a
#: new TLS handshake on the next one.
_KEEPALIVE_EXPIRY = 240.0

#: The decision model to ask.
DEFAULT_MODEL = "gemini-decision-1"
#: Seconds to wait for an answer, retries included, before giving up.
DEFAULT_TIMEOUT = 10.0


class GoogleDecisionsUsage(BaseModel):
    """Tokens the Decisions API used: for one request, or over every request.

    :attr:`GoogleDecisionsClient.usage` holds the total over every request.

    Parameters:
        input_tokens: Tokens sent.
        output_tokens: Tokens received.
    """

    input_tokens: int = 0
    output_tokens: int = 0


class GoogleDecisionsClient:
    """Client for Google's Decisions API, through ``google-genai``.

    Gives the ``google-genai`` client an HTTP/2 connection pool of its own,
    so many small requests share a connection and can be in flight at the
    same time. The connection is kept open between questions, so a gap
    between them does not cost a new TLS handshake. Retries with backoff
    when Google is busy or can't be reached, within one time bound for the
    whole request. Counts the tokens every request used in :attr:`usage`.
    Errors carry the reason Google gave.

    :meth:`connect` opens the connection ahead of the first question, and
    :meth:`close` releases it.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 3,
    ):
        """Initialize the client.

        Args:
            api_key: Gemini API key.
            model: The decision model to ask.
            timeout: Seconds to wait for an answer, retries included, before
                giving up.
            max_retries: How many times to retry a request that failed for a
                reason that may pass, such as Google being busy.
        """
        if not api_key:
            raise ValueError("GoogleDecisionsClient needs an API key")
        self._model = model
        self._timeout = timeout
        self._usage = GoogleDecisionsUsage()
        self._connected = False
        self._connect_lock = asyncio.Lock()
        self._http = httpx.AsyncClient(
            timeout=timeout,
            http2=True,
            limits=httpx.Limits(keepalive_expiry=_KEEPALIVE_EXPIRY),
        )
        self._genai = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                # The Decisions API is served on v1beta only.
                api_version="v1beta",
                httpx_async_client=self._http,
                timeout=int(timeout * 1000),
                # The attempts count the first request too.
                retry_options=types.HttpRetryOptions(
                    attempts=max_retries + 1, initial_delay=0.25, max_delay=2.0
                ),
            ),
        )

    @property
    def model(self) -> str:
        """The decision model the questions go to."""
        return self._model

    @property
    def usage(self) -> GoogleDecisionsUsage:
        """Tokens used so far, over every request."""
        return self._usage

    async def connect(self):
        """Open the connection to Google.

        Looks up the model, a cheap request that also checks the key and
        whether it has access to the model. Connects once: a client shared by
        several classifiers is asked by each of them, and only the first call
        sends anything.

        Raises:
            ClassifierError: If Google could not be reached or refused the
                request.
        """
        async with self._connect_lock:
            if self._connected:
                return
            try:
                await asyncio.wait_for(self._genai.aio.models.get(model=self._model), self._timeout)
            except TimeoutError as e:
                raise ClassifierError(
                    f"Google Decisions did not answer within {self._timeout:g}s"
                ) from e
            except Exception as e:
                raise ClassifierError(
                    f"could not connect to Google Decisions: {self._failure(e)}"
                ) from e
            self._connected = True

    async def close(self):
        """Close the connection pool."""
        self._connected = False
        self._genai.close()
        await self._http.aclose()

    async def ask(
        self, state: str | list[dict[str, Any]], questions: Mapping[str, dict[str, Any]]
    ) -> tuple[dict[str, Answer], GoogleDecisionsUsage]:
        """Send questions about one state and return Google's answers.

        Args:
            state: What the questions are about, as the request's ``input``:
                text, or content blocks.
            questions: The questions by name, each in Google's own format: a
                ``type`` of ``noul``, ``choice`` or ``score``,
                ``instructions``, and ``criteria``.

        Returns:
            The answers by the same names, as ``google-genai`` answers, and
            the tokens the request used.

        Raises:
            ClassifierError: If Google rejected the request, kept refusing it
                because it was busy, could not be reached, did not answer in
                time, blocked the input, or left a question unanswered.
        """
        try:
            # The state and questions come as plain dicts in Google's format,
            # which google-genai validates as it builds the request.
            response = await asyncio.wait_for(
                self._genai.aio.decisions.create(
                    model=self._model,
                    input=cast(InputParam, state),
                    questions={
                        name: cast(QuestionParam, question) for name, question in questions.items()
                    },
                ),
                self._timeout,
            )
        except TimeoutError as e:
            raise ClassifierError(
                f"Google Decisions did not answer within {self._timeout:g}s"
            ) from e
        except Exception as e:
            # google-genai raises errors of its own, and a request it can't
            # build as a pydantic ValidationError.
            raise ClassifierError(f"Google Decisions request failed: {self._failure(e)}") from e
        request_usage = GoogleDecisionsUsage(
            input_tokens=getattr(response.usage, "total_input_tokens", None) or 0,
            output_tokens=getattr(response.usage, "total_output_tokens", None) or 0,
        )
        self._usage.input_tokens += request_usage.input_tokens
        self._usage.output_tokens += request_usage.output_tokens
        if response.answers is None:
            raise ClassifierError(self._no_answers(response.prompt_feedback))
        missing = [name for name in questions if name not in response.answers]
        if missing:
            raise ClassifierError(f"Google Decisions reply has no answer for {', '.join(missing)}")
        return {name: response.answers[name] for name in questions}, request_usage

    def _no_answers(self, feedback: Any) -> str:
        """Why a reply has no answers: the reason Google's filters blocked the input, if they did."""
        reason = getattr(feedback, "block_reason", None)
        if not reason:
            return "Google Decisions reply has no answers"
        message = getattr(feedback, "block_reason_message", None)
        return f"Google Decisions blocked the input: {reason}" + (f": {message}" if message else "")

    def _failure(self, error: Exception) -> str:
        """The HTTP status and the reason Google gave for a failed request, or the error itself."""
        status = getattr(error, "status_code", None) or getattr(error, "code", None)
        if not status:
            return repr(error)
        body = getattr(error, "body", None) or getattr(error, "details", None)
        details = body.get("error") if isinstance(body, dict) else None
        message = str(details.get("message") or "").strip() if isinstance(details, dict) else ""
        return f"HTTP {status}: {message}" if message else f"HTTP {status}"
