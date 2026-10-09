#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
from collections.abc import Callable

import httpx
import pytest

from pipecat.classifiers.base_classifier import ClassifierError
from pipecat.classifiers.openai.decisions.client import OpenAIDecisionsClient

USAGE = {
    "input_tokens": 12,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": 0,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 12,
}
PREDICATE = {"answer": {"type": "predicate", "instructions": "?"}}


def _client(handler: Callable[[httpx.Request], httpx.Response], **kwargs) -> OpenAIDecisionsClient:
    """A client whose requests ``handler`` answers instead of OpenAI."""
    client = OpenAIDecisionsClient(api_key="key", **kwargs)
    client._http._transport = httpx.MockTransport(handler)
    return client


def _reply(*answers: dict) -> httpx.Response:
    return httpx.Response(
        200, json={"model": "gpt-6-luna", "answers": list(answers), "usage": USAGE}
    )


def _predicate(probability: float, name: str = "answer") -> dict:
    return {"type": "predicate", "name": name, "probability": probability}


def _failure(status: int, message: str, code: str | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        json={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": code,
            }
        },
    )


def _model(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200, json={"id": "gpt-6-luna", "object": "model", "created": 0, "owned_by": "system"}
    )


class TestOpenAIDecisionsClient:
    @pytest.mark.asyncio
    async def test_sends_questions_to_the_decisions_endpoint_with_auth(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers["Authorization"]
            seen["body"] = json.loads(request.content)
            return _reply(_predicate(0.9))

        client = _client(handler)
        answers, usage = await client.ask(
            "hello", {"answer": {"type": "predicate", "instructions": "a greeting?"}}
        )

        assert answers["answer"] == _predicate(0.9)
        assert (usage.input_tokens, usage.output_tokens) == (12, 0)
        assert seen["url"] == "https://api.openai.com/v1/decisions"
        assert seen["auth"] == "Bearer key"
        assert seen["body"] == {
            "model": "gpt-6-luna",
            "input": "hello",
            "questions": [{"name": "answer", "type": "predicate", "instructions": "a greeting?"}],
        }
        await client.close()

    @pytest.mark.asyncio
    async def test_the_base_url_and_model_pick_where_questions_go(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["model"] = json.loads(request.content)["model"]
            return _reply(_predicate(0.9))

        client = _client(handler, base_url="https://eu.api.openai.com/v1", model="gpt-6-sol")
        await client.ask("a", PREDICATE)

        assert seen == {"url": "https://eu.api.openai.com/v1/decisions", "model": "gpt-6-sol"}
        await client.close()

    @pytest.mark.asyncio
    async def test_answers_are_matched_to_questions_by_name(self):
        client = _client(lambda request: _reply(_predicate(0.2, "b"), _predicate(0.8, "a")))
        answers, _ = await client.ask(
            "x",
            {
                "a": {"type": "predicate", "instructions": "a?"},
                "b": {"type": "predicate", "instructions": "b?"},
            },
        )

        assert answers["a"]["probability"] == 0.8
        assert answers["b"]["probability"] == 0.2
        await client.close()

    @pytest.mark.asyncio
    async def test_counts_tokens_over_requests(self):
        client = _client(lambda request: _reply(_predicate(0.5)))
        await client.ask("a", PREDICATE)
        await client.ask("b", PREDICATE)

        assert client.usage.input_tokens == 24
        await client.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("busy", [429, 503])
    async def test_retries_when_busy(self, monkeypatch, busy):
        statuses = iter([busy, busy])
        waits = []

        async def no_sleep(seconds):
            waits.append(seconds)

        monkeypatch.setattr("pipecat.classifiers.openai.decisions.client.asyncio.sleep", no_sleep)

        def handler(request: httpx.Request) -> httpx.Response:
            status = next(statuses, None)
            if status is not None:
                return _failure(status, "Rate limit reached")
            return _reply(_predicate(0.7))

        client = _client(handler)
        answers, _ = await client.ask("a", PREDICATE)

        assert answers["answer"]["probability"] == 0.7
        assert waits == [0.25, 0.5]
        await client.close()

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries_with_openais_reason(self, monkeypatch):
        async def no_sleep(seconds):
            pass

        monkeypatch.setattr("pipecat.classifiers.openai.decisions.client.asyncio.sleep", no_sleep)
        client = _client(lambda request: _failure(429, "Rate limit reached"), max_retries=2)

        with pytest.raises(ClassifierError, match=r"busy \(HTTP 429: Rate limit reached\)"):
            await client.ask("a", PREDICATE)
        await client.close()

    @pytest.mark.asyncio
    async def test_a_rejected_request_carries_openais_reason(self):
        client = _client(
            lambda request: _failure(
                400,
                "Invalid type for 'questions[0].instructions': expected a string",
                "invalid_type",
            )
        )

        with pytest.raises(
            ClassifierError,
            match=r"rejected the request: HTTP 400: Invalid type for 'questions\[0\]",
        ):
            await client.ask("a", PREDICATE)
        await client.close()

    @pytest.mark.asyncio
    async def test_a_rejected_request_without_a_reason_has_only_the_status(self):
        client = _client(lambda request: httpx.Response(502, text="<html>bad gateway</html>"))

        with pytest.raises(ClassifierError, match=r"rejected the request: HTTP 502$"):
            await client.ask("a", PREDICATE)
        await client.close()

    @pytest.mark.asyncio
    async def test_a_reply_that_is_not_json_is_an_error(self):
        client = _client(lambda request: httpx.Response(200, text="<html>busy</html>"))

        with pytest.raises(ClassifierError, match="not valid JSON"):
            await client.ask("a", PREDICATE)
        await client.close()

    @pytest.mark.asyncio
    async def test_a_reply_that_is_not_an_object_is_an_error(self):
        client = _client(lambda request: httpx.Response(200, json=[]))

        with pytest.raises(ClassifierError, match="not a JSON object"):
            await client.ask("a", PREDICATE)
        await client.close()

    @pytest.mark.asyncio
    async def test_a_reply_without_answers_is_an_error(self):
        client = _client(lambda request: httpx.Response(200, json={"model": "gpt-6-luna"}))

        with pytest.raises(ClassifierError, match="no answers"):
            await client.ask("a", PREDICATE)
        await client.close()

    @pytest.mark.asyncio
    async def test_a_question_left_unanswered_is_an_error(self):
        client = _client(lambda request: _reply(_predicate(0.5, "other"), {"name": None}))

        with pytest.raises(ClassifierError, match="no answer for answer"):
            await client.ask("a", PREDICATE)
        await client.close()

    @pytest.mark.asyncio
    async def test_usage_that_is_not_an_object_is_ignored(self):
        client = _client(
            lambda request: httpx.Response(
                200, json={"answers": [_predicate(0.5)], "usage": "none"}
            )
        )
        await client.ask("a", PREDICATE)

        assert client.usage.input_tokens == 0
        await client.close()

    @pytest.mark.asyncio
    async def test_unreachable_is_an_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        client = _client(handler)

        with pytest.raises(ClassifierError, match="failed"):
            await client.ask("a", PREDICATE)
        await client.close()

    @pytest.mark.asyncio
    async def test_connect_looks_up_the_model(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((request.method, request.url.path))
            return _model(request)

        client = _client(handler, model="gpt-6-sol")
        await client.connect()

        assert seen == [("GET", "/v1/models/gpt-6-sol")]
        await client.close()

    @pytest.mark.asyncio
    async def test_connect_sends_only_once(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.path)
            return _model(request)

        client = _client(handler)
        await client.connect()
        await client.connect()

        assert len(seen) == 1
        await client.close()

    @pytest.mark.asyncio
    async def test_a_refused_connect_carries_openais_reason(self):
        client = _client(
            lambda request: _failure(
                404,
                "The model `gpt-6-luna` does not exist or you do not have access to it.",
                "model_not_found",
            )
        )

        with pytest.raises(ClassifierError, match="HTTP 404: The model `gpt-6-luna` does not"):
            await client.connect()
        await client.close()

    @pytest.mark.asyncio
    async def test_keeps_idle_connections_open_between_questions(self):
        client = OpenAIDecisionsClient(api_key="key")
        assert client._http._transport._pool._keepalive_expiry == 240.0
        await client.close()

    def test_needs_an_api_key(self):
        with pytest.raises(ValueError):
            OpenAIDecisionsClient(api_key="")
