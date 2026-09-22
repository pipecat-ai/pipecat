#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
import os
from collections.abc import Callable

import httpx
import pytest

from pipecat.classifiers.base_classifier import (
    ChoiceQuestion,
    ClassifierError,
    ScoreQuestion,
    YesNoQuestion,
)
from pipecat.classifiers.jev import JevClassifier
from pipecat.classifiers.jev_client import JevClient
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.workers.base_worker import BaseWorker

USAGE = {"input_tokens": 12, "output_tokens": 3}


def _client(handler: Callable[[httpx.Request], httpx.Response], **kwargs) -> JevClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return JevClient(api_key="key", http_client=http, **kwargs)


def _owner() -> BaseWorker:
    """A worker for a classifier to be set up in."""
    return BaseWorker("owner", task_manager=TaskManager())


def _reply(answer: dict) -> httpx.Response:
    return httpx.Response(
        200, json={"model": "jev-latest", "answers": {"answer": answer}, "usage": USAGE}
    )


class TestJevClient:
    @pytest.mark.asyncio
    async def test_sends_one_question_with_auth(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers["Authorization"]
            seen["body"] = json.loads(request.content)
            return _reply({"type": "noul", "noul": 0.9})

        client = _client(handler)
        answers = await client.ask(
            "hello", {"answer": {"type": "noul", "instructions": "a greeting?"}}
        )
        answer = answers["answer"]

        assert answer == {"type": "noul", "noul": 0.9}
        assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
        assert seen["auth"] == "Bearer key"
        assert seen["body"] == {
            "model": "jev-latest",
            "state": "hello",
            "questions": {"answer": {"type": "noul", "instructions": "a greeting?"}},
        }
        await client.close()

    @pytest.mark.asyncio
    async def test_counts_tokens_over_requests(self):
        client = _client(lambda request: _reply({"type": "noul", "noul": 0.5}))
        await client.ask("a", {"answer": {"type": "noul", "instructions": "?"}})
        await client.ask("b", {"answer": {"type": "noul", "instructions": "?"}})

        assert client.usage.input_tokens == 24
        assert client.usage.output_tokens == 6
        await client.close()

    @pytest.mark.asyncio
    async def test_retries_when_busy(self, monkeypatch):
        statuses = iter([429, 529])
        waits = []

        async def no_sleep(seconds):
            waits.append(seconds)

        monkeypatch.setattr("pipecat.classifiers.jev_client.asyncio.sleep", no_sleep)

        def handler(request: httpx.Request) -> httpx.Response:
            status = next(statuses, None)
            if status is not None:
                return httpx.Response(status)
            return _reply({"type": "noul", "noul": 0.7})

        client = _client(handler)
        answer = (await client.ask("a", {"answer": {"type": "noul", "instructions": "?"}}))[
            "answer"
        ]

        assert answer["noul"] == 0.7
        assert waits == [0.25, 0.5]
        await client.close()

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self, monkeypatch):
        async def no_sleep(seconds):
            pass

        monkeypatch.setattr("pipecat.classifiers.jev_client.asyncio.sleep", no_sleep)
        client = _client(lambda request: httpx.Response(429), max_retries=2)

        with pytest.raises(ClassifierError, match="busy"):
            await client.ask("a", {"answer": {"type": "noul", "instructions": "?"}})
        await client.close()

    @pytest.mark.asyncio
    async def test_rejected_request_is_an_error(self):
        client = _client(lambda request: httpx.Response(401))

        with pytest.raises(ClassifierError, match="401"):
            await client.ask("a", {"answer": {"type": "noul", "instructions": "?"}})
        await client.close()

    @pytest.mark.asyncio
    async def test_unreachable_is_an_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        client = _client(handler)

        with pytest.raises(ClassifierError, match="failed"):
            await client.ask("a", {"answer": {"type": "noul", "instructions": "?"}})
        await client.close()

    def test_needs_an_api_key(self):
        with pytest.raises(ValueError):
            JevClient(api_key="")


class TestJevClassifier:
    @pytest.mark.asyncio
    async def test_yes_no(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["question"] = json.loads(request.content)["questions"]["answer"]
            return _reply({"type": "noul", "noul": 0.95})

        classifier = JevClassifier(client=_client(handler))
        result = (
            await classifier.yes_no(
                "Please leave a message",
                {"answer": YesNoQuestion(instructions="is this a voicemail greeting?")},
            )
        )["answer"]

        assert result.probability == 0.95
        assert seen["question"] == {"type": "noul", "instructions": "is this a voicemail greeting?"}
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_yes_no_with_what_counts_as_each_answer(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["question"] = json.loads(request.content)["questions"]["answer"]
            return _reply({"type": "noul", "noul": 0.2})

        classifier = JevClassifier(client=_client(handler))
        (
            await classifier.yes_no(
                "Hello?",
                {
                    "answer": YesNoQuestion(
                        instructions="is this a voicemail greeting?",
                        yes="a recorded greeting or carrier message",
                        no="a person talking",
                    )
                },
            )
        )["answer"]

        assert seen["question"] == {
            "type": "noul",
            "instructions": "is this a voicemail greeting?",
            "criteria": {
                "true": "a recorded greeting or carrier message",
                "false": "a person talking",
            },
        }
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_choice(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["question"] = json.loads(request.content)["questions"]["answer"]
            return _reply(
                {
                    "type": "choice",
                    "choice": "short",
                    "probabilities": {"complete": 0.1, "short": 0.88},
                    "confidence": 0.8,
                }
            )

        classifier = JevClassifier(client=_client(handler))
        options = {
            "complete": "the turn is over",
            "short": "a brief pause",
            "long": "asked for time",
        }
        result = (
            await classifier.choice(
                "I think, um",
                {
                    "answer": ChoiceQuestion(
                        instructions="is the user's turn over?", options=options
                    )
                },
            )
        )["answer"]

        assert result.label == "short"
        assert result.probabilities == {"complete": 0.1, "short": 0.88, "long": 0.0}
        assert result.confidence == 0.8
        assert seen["question"] == {
            "type": "choice",
            "instructions": "is the user's turn over?",
            "criteria": options,
        }
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_score_keys_probabilities_by_level(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["question"] = json.loads(request.content)["questions"]["answer"]
            return _reply(
                {
                    "type": "score",
                    "score": 1.05,
                    "legend": {"0": "calm", "1": "frustrated", "2": "angry"},
                    "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
                    "confidence": 0.92,
                }
            )

        classifier = JevClassifier(client=_client(handler))
        rubric = ["calm", "frustrated", "angry"]
        result = (
            await classifier.score(
                "This is the third time!",
                {"answer": ScoreQuestion(instructions="how upset is the user?", rubric=rubric)},
            )
        )["answer"]

        assert result.score == 1.05
        assert result.probabilities == {"calm": 0.0, "frustrated": 0.95, "angry": 0.05}
        assert result.confidence == 0.92
        assert seen["question"] == {
            "type": "score",
            "instructions": "how upset is the user?",
            "criteria": rubric,
        }
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_missing_answer_field_is_an_error(self):
        classifier = JevClassifier(client=_client(lambda request: _reply({"type": "noul"})))

        with pytest.raises(ClassifierError, match="noul"):
            (await classifier.yes_no("a", {"answer": YesNoQuestion(instructions="?")}))["answer"]
        await classifier.client.close()

    def test_needs_a_key_or_a_client(self):
        with pytest.raises(ValueError):
            JevClassifier()

    @pytest.mark.asyncio
    async def test_cleanup_closes_only_an_owned_client(self):
        owned = JevClassifier(api_key="key")
        await owned.cleanup()
        assert owned.client._http.is_closed

        shared = JevClient(api_key="key")
        classifier = JevClassifier(client=shared)
        await classifier.cleanup()
        assert not shared._http.is_closed
        await shared.close()


@pytest.mark.skipif(not os.getenv("TYPESAFE_API_KEY"), reason="TYPESAFE_API_KEY not set")
class TestJevLive:
    @pytest.mark.asyncio
    async def test_three_questions(self):
        classifier = JevClassifier(api_key=os.environ["TYPESAFE_API_KEY"])
        try:
            greeting = "Hi, you've reached Sam. I can't take your call right now, leave a message."
            yes_no = (
                await classifier.yes_no(
                    greeting,
                    {"answer": YesNoQuestion(instructions="is this a voicemail greeting?")},
                )
            )["answer"]
            assert yes_no.probability > 0.5

            choice = (
                await classifier.choice(
                    "I'd like to book a table for, um",
                    {
                        "answer": ChoiceQuestion(
                            instructions="is the user's turn over?",
                            options={
                                "complete": "the user finished",
                                "short": "a brief pause",
                                "long": "asked for time",
                            },
                        )
                    },
                )
            )["answer"]
            assert choice.label in ("complete", "short", "long")
            assert abs(sum(choice.probabilities.values()) - 1.0) < 0.05

            score = (
                await classifier.score(
                    "This is the third time I call and nobody helps me!",
                    {
                        "answer": ScoreQuestion(
                            instructions="how upset is the user?",
                            rubric=["calm", "impatient", "frustrated", "asking for a person"],
                        )
                    },
                )
            )["answer"]
            assert 0 <= score.score <= 3
            assert classifier.client.usage.input_tokens > 0
        finally:
            await classifier.cleanup()


class TestJevClientConnection:
    @pytest.mark.asyncio
    async def test_keeps_idle_connections_open_between_questions(self):
        client = JevClient(api_key="key")
        transport = client._http._transport
        assert transport._pool._keepalive_expiry == 240.0
        await client.close()

    @pytest.mark.asyncio
    async def test_setup_opens_the_connection(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((request.method, request.url.path))
            return httpx.Response(200, json={"models": []})

        classifier = JevClassifier(client=_client(handler))
        await classifier.setup(_owner())

        assert seen == [("GET", "/v1/models")]
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_a_failed_connect_at_setup_is_only_a_warning(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        classifier = JevClassifier(client=_client(handler))
        await classifier.setup(_owner())
        await classifier.client.close()


class TestJevClassifierStructuredQuestions:
    @pytest.mark.asyncio
    async def test_choice_with_structured_and_empty_descriptions(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["question"] = json.loads(request.content)["questions"]["answer"]
            return _reply(
                {"type": "choice", "choice": "b", "probabilities": {"b": 1.0}, "confidence": 1.0}
            )

        classifier = JevClassifier(client=_client(handler))
        options = {"a": {"examples": ["hi", "hello"]}, "b": None}
        result = (
            await classifier.choice(
                "hey",
                {
                    "answer": ChoiceQuestion(
                        instructions={"question": "which?", "note": "casual"}, options=options
                    )
                },
            )
        )["answer"]

        assert result.label == "b"
        assert seen["question"] == {
            "type": "choice",
            "instructions": {"question": "which?", "note": "casual"},
            "criteria": options,
        }
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_score_with_structured_levels_keys_probabilities_by_their_json(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return _reply(
                {
                    "type": "score",
                    "score": 1.0,
                    "probabilities": {"0": 0.1, "1": 0.9},
                    "confidence": 0.9,
                }
            )

        classifier = JevClassifier(client=_client(handler))
        rubric = [{"level": "calm"}, {"level": "angry", "signs": ["shouting"]}]
        result = (
            await classifier.score(
                "This is the third time!",
                {"answer": ScoreQuestion(instructions="how upset?", rubric=rubric)},
            )
        )["answer"]

        assert result.probabilities == {
            '{"level": "calm"}': 0.1,
            '{"level": "angry", "signs": ["shouting"]}': 0.9,
        }
        await classifier.client.close()


class TestJevClassifierSeveralQuestions:
    @pytest.mark.asyncio
    async def test_ask_sends_every_question_in_one_request(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "model": "jev-latest",
                    "answers": {
                        "greeting": {"type": "noul", "noul": 0.9},
                        "mood": {
                            "type": "score",
                            "score": 0.2,
                            "probabilities": {"0": 0.8, "1": 0.2},
                            "confidence": 0.8,
                        },
                    },
                    "usage": USAGE,
                },
            )

        classifier = JevClassifier(client=_client(handler))
        results = await classifier.ask(
            "Hello there!",
            {
                "greeting": YesNoQuestion(instructions="is this a greeting?"),
                "mood": ScoreQuestion(instructions="how upset?", rubric=["calm", "upset"]),
            },
        )

        assert set(seen["body"]["questions"]) == {"greeting", "mood"}
        assert results["greeting"].probability == 0.9
        assert results["mood"].score == 0.2
        assert results["mood"].probabilities == {"calm": 0.8, "upset": 0.2}
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_typed_method_takes_several_questions_of_its_kind(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "answers": {
                        "a": {"type": "noul", "noul": 0.1},
                        "b": {"type": "noul", "noul": 0.7},
                    },
                    "usage": USAGE,
                },
            )

        classifier = JevClassifier(client=_client(handler))
        results = await classifier.yes_no(
            "hi", {"a": YesNoQuestion(instructions="a?"), "b": YesNoQuestion(instructions="b?")}
        )

        assert {name: r.probability for name, r in results.items()} == {"a": 0.1, "b": 0.7}
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_a_missing_answer_is_an_error(self):
        classifier = JevClassifier(
            client=_client(lambda request: _reply({"type": "noul", "noul": 0.5}))
        )
        with pytest.raises(ClassifierError, match="no answer for other"):
            await classifier.ask(
                "hi",
                {
                    "answer": YesNoQuestion(instructions="?"),
                    "other": YesNoQuestion(instructions="?"),
                },
            )
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_a_shared_client_connects_once(self):
        connects = []

        def handler(request: httpx.Request) -> httpx.Response:
            connects.append(request.url.path)
            return httpx.Response(200, json={"models": []})

        client = _client(handler)
        first, second = JevClassifier(client=client), JevClassifier(client=client)
        owner = _owner()
        await asyncio.gather(first.setup(owner), second.setup(owner))
        await first.setup(owner)

        assert connects == ["/v1/models"]
        await client.close()
