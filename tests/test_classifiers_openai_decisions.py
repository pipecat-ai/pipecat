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
from pipecat.classifiers.openai.decisions.classifier import (
    OPENAI_DECISIONS_MAX_CHOICE_OPTIONS,
    OpenAIDecisionsClassifier,
)
from pipecat.classifiers.openai.decisions.client import OpenAIDecisionsClient
from pipecat.metrics.metrics import LLMUsageMetricsData, ProcessingMetricsData
from pipecat.utils.asyncio.task_manager import TaskManager

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


def _question(request: httpx.Request) -> dict:
    """The one question a request sent, without its name."""
    (question,) = json.loads(request.content)["questions"]
    return {key: value for key, value in question.items() if key != "name"}


class TestOpenAIDecisionsClassifier:
    @pytest.mark.asyncio
    async def test_yes_no(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["question"] = _question(request)
            return _reply(_predicate(0.95))

        classifier = OpenAIDecisionsClassifier(client=_client(handler))
        result = (
            await classifier.yes_no(
                "Please leave a message",
                {"answer": YesNoQuestion(instructions="is this a voicemail greeting?")},
            )
        )["answer"]

        assert result.probability == 0.95
        assert seen["question"] == {
            "type": "predicate",
            "instructions": "is this a voicemail greeting?",
        }
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_yes_no_says_what_counts_as_each_answer_in_the_instructions(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["question"] = _question(request)
            return _reply(_predicate(0.2))

        classifier = OpenAIDecisionsClassifier(client=_client(handler))
        await classifier.yes_no(
            "Hello?",
            {
                "answer": YesNoQuestion(
                    instructions="is this a voicemail greeting?",
                    yes="a recorded greeting",
                    no="a person talking",
                )
            },
        )

        assert seen["question"]["instructions"] == (
            "is this a voicemail greeting?\n"
            "Yes means: a recorded greeting\n"
            "No means: a person talking"
        )
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_structured_state_and_instructions_are_sent_as_json(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen["input"] = body["input"]
            seen["instructions"] = body["questions"][0]["instructions"]
            return _reply(_predicate(0.5))

        classifier = OpenAIDecisionsClassifier(client=_client(handler))
        state = [{"role": "user", "content": "hi"}]
        instructions = {"question": "did they greet?", "speaker": "user"}
        await classifier.yes_no(state, {"answer": YesNoQuestion(instructions=instructions)})

        assert json.loads(seen["input"]) == state
        assert json.loads(seen["instructions"]) == instructions
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_choice(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["question"] = _question(request)
            return _reply(
                {
                    "type": "choice",
                    "name": "answer",
                    "choice": "technical",
                    "probabilities": [
                        {"value": "billing", "probability": 0.16},
                        {"value": "technical", "probability": 0.83},
                    ],
                    "confidence": 0.56,
                }
            )

        classifier = OpenAIDecisionsClassifier(client=_client(handler))
        options = {"billing": "payments", "technical": {"covers": ["outages"]}, "sales": None}
        result = (
            await classifier.choice(
                "Checkout is down",
                {"answer": ChoiceQuestion(instructions="which team?", options=options)},
            )
        )["answer"]

        assert result.choice == "technical"
        assert result.probabilities == {"billing": 0.16, "technical": 0.83, "sales": 0.0}
        assert result.confidence == 0.56
        assert seen["question"] == {
            "type": "choice",
            "instructions": "which team?",
            "choices": [
                {"value": "billing", "description": "payments"},
                {"value": "technical", "description": '{\n  "covers": [\n    "outages"\n  ]\n}'},
                {"value": "sales"},
            ],
        }
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_too_many_options_is_an_error_before_any_request(self):
        sent = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            return _reply(_predicate(1.0))

        classifier = OpenAIDecisionsClassifier(client=_client(handler))
        options = {f"o{i}": None for i in range(OPENAI_DECISIONS_MAX_CHOICE_OPTIONS + 1)}
        with pytest.raises(ClassifierError, match="at most 255 options"):
            await classifier.choice(
                "hmm", {"answer": ChoiceQuestion(instructions="?", options=options)}
            )
        assert sent == []
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_a_choice_outside_the_options_is_an_error(self):
        classifier = OpenAIDecisionsClassifier(
            client=_client(
                lambda request: _reply(
                    {"type": "choice", "name": "answer", "choice": "maybe", "confidence": 0.8}
                )
            )
        )
        question = ChoiceQuestion(instructions="?", options={"yes": None, "no": None})
        with pytest.raises(ClassifierError, match="OpenAI Decisions chose 'maybe'"):
            await classifier.choice("hmm", {"answer": question})
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_an_unusable_probability_is_an_error(self):
        classifier = OpenAIDecisionsClassifier(
            client=_client(
                lambda request: _reply(
                    {
                        "type": "choice",
                        "name": "answer",
                        "choice": "yes",
                        "probabilities": [
                            {"value": "yes", "probability": "high"},
                            {"value": "no", "probability": 0.1},
                        ],
                        "confidence": 0.8,
                    }
                )
            )
        )
        question = ChoiceQuestion(instructions="?", options={"yes": None, "no": None})
        with pytest.raises(ClassifierError, match="unusable probability"):
            await classifier.choice("hmm", {"answer": question})
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_score_keys_probabilities_by_level(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["question"] = _question(request)
            return _reply(
                {
                    "type": "score",
                    "name": "answer",
                    "score": 1.97,
                    "probabilities": [
                        {"value": 0, "label": "calm", "probability": 0.01},
                        {"value": 1, "label": "impatient", "probability": 0.01},
                        {"value": 2, "label": '{\n  "level": "angry"\n}', "probability": 0.98},
                    ],
                    "confidence": 0.96,
                }
            )

        classifier = OpenAIDecisionsClassifier(client=_client(handler))
        levels = ["calm", "impatient", {"level": "angry"}]
        result = (
            await classifier.score(
                "This is the third time!",
                {"answer": ScoreQuestion(instructions="how upset?", levels=levels)},
            )
        )["answer"]

        assert result.score == 1.97
        assert [(l.level, l.probability) for l in result.levels] == [
            ("calm", 0.01),
            ("impatient", 0.01),
            ({"level": "angry"}, 0.98),
        ]
        assert result.confidence == 0.96
        assert seen["question"] == {
            "type": "score",
            "instructions": "how upset?",
            "levels": [
                {"label": "calm"},
                {"label": "impatient"},
                {"label": '{\n  "level": "angry"\n}'},
            ],
        }
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_a_missing_answer_field_is_an_error(self):
        classifier = OpenAIDecisionsClassifier(
            client=_client(lambda request: _reply({"type": "predicate", "name": "answer"}))
        )

        with pytest.raises(ClassifierError, match="probability"):
            await classifier.yes_no("a", {"answer": YesNoQuestion(instructions="?")})
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_a_refusal_is_an_error_naming_the_question(self):
        classifier = OpenAIDecisionsClassifier(
            client=_client(lambda request: _reply({"type": "refusal", "name": "answer"}))
        )

        with pytest.raises(ClassifierError, match="refused to answer 'answer'"):
            await classifier.yes_no("a", {"answer": YesNoQuestion(instructions="?")})
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_ask_sends_every_question_in_one_request(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return _reply(
                _predicate(0.9, "greeting"),
                {
                    "type": "score",
                    "name": "mood",
                    "score": 0.2,
                    "probabilities": [
                        {"value": 0, "label": "calm", "probability": 0.8},
                        {"value": 1, "label": "upset", "probability": 0.2},
                    ],
                    "confidence": 0.8,
                },
            )

        classifier = OpenAIDecisionsClassifier(client=_client(handler))
        results = await classifier.ask(
            "Hello there!",
            {
                "greeting": YesNoQuestion(instructions="is this a greeting?"),
                "mood": ScoreQuestion(instructions="how upset?", levels=["calm", "upset"]),
            },
        )

        assert len(requests) == 1
        assert [q["name"] for q in requests[0]["questions"]] == ["greeting", "mood"]
        assert results["greeting"].probability == 0.9
        assert results["mood"].probability("calm") == 0.8
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_a_missing_answer_is_an_error(self):
        classifier = OpenAIDecisionsClassifier(
            client=_client(lambda request: _reply(_predicate(0.5)))
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

    def test_needs_an_api_key_or_a_client(self):
        with pytest.raises(ValueError):
            OpenAIDecisionsClassifier()

    def test_a_client_of_its_own_takes_the_model_base_url_and_timeout(self):
        classifier = OpenAIDecisionsClassifier(
            api_key="key", model="gpt-6-sol", base_url="https://eu.api.openai.com/v1", timeout=1.5
        )
        assert classifier._owns_client
        assert classifier.model == "gpt-6-sol"
        assert str(classifier.client._http.base_url) == "https://eu.api.openai.com/v1/"
        assert classifier.client._http.timeout.read == 1.5

    @pytest.mark.asyncio
    async def test_cleanup_closes_only_an_owned_client(self):
        owned = OpenAIDecisionsClassifier(api_key="key")
        await owned.cleanup()
        assert owned.client._http.is_closed

        shared = OpenAIDecisionsClient(api_key="key")
        classifier = OpenAIDecisionsClassifier(client=shared)
        await classifier.cleanup()
        assert not shared._http.is_closed
        await shared.close()

    @pytest.mark.asyncio
    async def test_setup_opens_the_connection(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.path)
            return _model(request)

        classifier = OpenAIDecisionsClassifier(client=_client(handler))
        await classifier.setup(TaskManager())

        assert seen == ["/v1/models/gpt-6-luna"]
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_a_failed_connect_at_setup_is_only_a_warning(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        classifier = OpenAIDecisionsClassifier(client=_client(handler))
        await classifier.setup(TaskManager())
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_a_shared_client_connects_once(self):
        connects = []

        def handler(request: httpx.Request) -> httpx.Response:
            connects.append(request.url.path)
            return _model(request)

        client = _client(handler)
        first = OpenAIDecisionsClassifier(client=client)
        second = OpenAIDecisionsClassifier(client=client)
        task_manager = TaskManager()
        await asyncio.gather(first.setup(task_manager), second.setup(task_manager))
        await first.setup(task_manager)

        assert len(connects) == 1
        await client.close()

    @pytest.mark.asyncio
    async def test_every_call_reports_its_time_and_tokens(self):
        client = _client(lambda request: _reply(_predicate(0.9)))
        classifier = OpenAIDecisionsClassifier(client=client, name="voicemail")
        reported = asyncio.Event()
        seen: list = []

        @classifier.event_handler("on_metrics")
        async def on_metrics(classifier, data):
            seen.extend(data)
            reported.set()

        await classifier.yes_no("hello", {"answer": YesNoQuestion(instructions="a greeting?")})
        await asyncio.wait_for(reported.wait(), 1)

        processing, usage = seen
        assert isinstance(processing, ProcessingMetricsData)
        assert (processing.processor, processing.model) == ("voicemail", "gpt-6-luna")
        assert isinstance(usage, LLMUsageMetricsData)
        assert (usage.value.prompt_tokens, usage.value.total_tokens) == (12, 12)
        await client.close()


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestOpenAIDecisionsLive:
    @pytest.mark.asyncio
    async def test_three_questions(self):
        classifier = OpenAIDecisionsClassifier(api_key=os.environ["OPENAI_API_KEY"])
        try:
            await classifier.client.connect()
            results = await classifier.ask(
                "Hi, you've reached Sam. I can't take your call right now, leave a message.",
                {
                    "voicemail": YesNoQuestion(instructions="is this a voicemail greeting?"),
                    "turn": ChoiceQuestion(
                        instructions="is the speaker's turn over?",
                        options={"complete": "finished", "short": "a brief pause"},
                    ),
                    "mood": ScoreQuestion(
                        instructions="how upset is the speaker?",
                        levels=["calm", "impatient", "angry"],
                    ),
                },
            )
            assert results["voicemail"].probability > 0.5
            assert results["turn"].choice in ("complete", "short")
            assert abs(sum(results["turn"].probabilities.values()) - 1.0) < 0.05
            assert 0 <= results["mood"].score <= 2
            assert classifier.client.usage.input_tokens > 0
        finally:
            await classifier.cleanup()
