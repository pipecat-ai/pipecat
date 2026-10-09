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
from pipecat.classifiers.cloudflare.clef.classifier import CLEF_MAX_CHOICE_OPTIONS, ClefClassifier
from pipecat.classifiers.cloudflare.clef.client import ClefClient
from pipecat.metrics.metrics import LLMUsageMetricsData, ProcessingMetricsData
from pipecat.utils.asyncio.task_manager import TaskManager

USAGE = {"input_tokens": 12, "output_tokens": 0}
NOUL = {"answer": {"type": "noul", "instructions": "?"}}


def _client(handler: Callable[[httpx.Request], httpx.Response], **kwargs) -> ClefClient:
    """A client whose requests ``handler`` answers instead of Workers AI."""
    client = ClefClient(account_id="acct", api_key="key", **kwargs)
    client._http._transport = httpx.MockTransport(handler)
    return client


def _success(result: dict) -> httpx.Response:
    return httpx.Response(
        200, json={"result": result, "success": True, "errors": [], "messages": []}
    )


def _reply(answer: dict) -> httpx.Response:
    return _success({"model": "clef", "answers": {"answer": answer}, "usage": USAGE})


def _failure(status: int, *reasons: str) -> httpx.Response:
    return httpx.Response(
        status,
        json={
            "result": None,
            "success": False,
            "errors": [{"code": 1000, "message": reason} for reason in reasons],
            "messages": [],
        },
    )


def _schema(request: httpx.Request) -> httpx.Response:
    return _success({"input": {}, "output": {}})


class TestClefClient:
    @pytest.mark.asyncio
    async def test_sends_questions_to_the_models_endpoint_with_auth(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers["Authorization"]
            seen["body"] = json.loads(request.content)
            return _reply({"type": "noul", "noul": 0.9})

        client = _client(handler)
        answers, usage = await client.ask(
            "hello", {"answer": {"type": "noul", "instructions": "a greeting?"}}
        )

        assert answers["answer"] == {"type": "noul", "noul": 0.9}
        assert (usage.input_tokens, usage.output_tokens) == (12, 0)
        assert seen["url"] == (
            "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef"
        )
        assert seen["auth"] == "Bearer key"
        assert seen["body"] == {
            "model": "clef",
            "state": "hello",
            "questions": {"answer": {"type": "noul", "instructions": "a greeting?"}},
        }
        await client.close()

    @pytest.mark.asyncio
    async def test_the_model_picks_the_endpoint(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["model"] = json.loads(request.content)["model"]
            return _reply({"type": "noul", "noul": 0.9})

        client = _client(handler, model="clef-flash")
        await client.ask("a", NOUL)

        assert seen == {
            "path": "/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef-flash",
            "model": "clef-flash",
        }
        await client.close()

    @pytest.mark.asyncio
    async def test_counts_tokens_over_requests(self):
        client = _client(lambda request: _reply({"type": "noul", "noul": 0.5}))
        await client.ask("a", NOUL)
        await client.ask("b", NOUL)

        assert client.usage.input_tokens == 24
        await client.close()

    @pytest.mark.asyncio
    async def test_retries_when_out_of_capacity(self, monkeypatch):
        statuses = iter([429, 429])
        waits = []

        async def no_sleep(seconds):
            waits.append(seconds)

        monkeypatch.setattr("pipecat.classifiers.cloudflare.clef.client.asyncio.sleep", no_sleep)

        def handler(request: httpx.Request) -> httpx.Response:
            status = next(statuses, None)
            if status is not None:
                return _failure(status, "Capacity temporarily exceeded, please try again")
            return _reply({"type": "noul", "noul": 0.7})

        client = _client(handler)
        answers, _ = await client.ask("a", NOUL)

        assert answers["answer"]["noul"] == 0.7
        assert waits == [0.25, 0.5]
        await client.close()

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries_with_cloudflares_reason(self, monkeypatch):
        async def no_sleep(seconds):
            pass

        monkeypatch.setattr("pipecat.classifiers.cloudflare.clef.client.asyncio.sleep", no_sleep)
        client = _client(lambda request: _failure(429, "Account limited"), max_retries=2)

        with pytest.raises(ClassifierError, match=r"busy \(HTTP 429: Account limited\)"):
            await client.ask("a", NOUL)
        await client.close()

    @pytest.mark.asyncio
    async def test_a_rejected_request_carries_cloudflares_reasons(self):
        client = _client(lambda request: _failure(403, "Account blocked", "Try later"))

        with pytest.raises(
            ClassifierError, match="Clef rejected the request: HTTP 403: Account blocked; Try later"
        ):
            await client.ask("a", NOUL)
        await client.close()

    @pytest.mark.asyncio
    async def test_a_rejected_request_without_reasons_has_only_the_status(self):
        client = _client(lambda request: httpx.Response(502, text="<html>bad gateway</html>"))

        with pytest.raises(ClassifierError, match=r"rejected the request: HTTP 502$"):
            await client.ask("a", NOUL)
        await client.close()

    @pytest.mark.asyncio
    async def test_a_reply_that_is_not_json_is_an_error(self):
        client = _client(lambda request: httpx.Response(200, text="<html>busy</html>"))

        with pytest.raises(ClassifierError, match="not valid JSON"):
            await client.ask("a", NOUL)
        await client.close()

    @pytest.mark.asyncio
    async def test_a_reply_without_a_result_is_an_error(self):
        client = _client(lambda request: httpx.Response(200, json={"success": True}))

        with pytest.raises(ClassifierError, match="no result"):
            await client.ask("a", NOUL)
        await client.close()

    @pytest.mark.asyncio
    async def test_a_reply_without_answers_is_an_error(self):
        client = _client(lambda request: _success({"model": "clef"}))

        with pytest.raises(ClassifierError, match="no answers"):
            await client.ask("a", NOUL)
        await client.close()

    @pytest.mark.asyncio
    async def test_usage_that_is_not_an_object_is_ignored(self):
        client = _client(
            lambda request: _success(
                {"answers": {"answer": {"type": "noul", "noul": 0.5}}, "usage": "none"}
            )
        )
        await client.ask("a", NOUL)

        assert client.usage.input_tokens == 0
        await client.close()

    @pytest.mark.asyncio
    async def test_unreachable_is_an_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        client = _client(handler)

        with pytest.raises(ClassifierError, match="failed"):
            await client.ask("a", NOUL)
        await client.close()

    @pytest.mark.asyncio
    async def test_connect_asks_for_the_models_schema(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((request.method, request.url.path, request.url.params["model"]))
            return _schema(request)

        client = _client(handler, model="clef-flash")
        await client.connect()

        assert seen == [
            ("GET", "/client/v4/accounts/acct/ai/models/schema", "@cf/cloudflare/clef-flash")
        ]
        await client.close()

    @pytest.mark.asyncio
    async def test_a_refused_connect_carries_cloudflares_reason(self):
        client = _client(lambda request: _failure(404, "Model schema not found"))

        with pytest.raises(ClassifierError, match="HTTP 404: Model schema not found"):
            await client.connect()
        await client.close()

    @pytest.mark.asyncio
    async def test_keeps_idle_connections_open_between_questions(self):
        client = ClefClient(account_id="acct", api_key="key")
        assert client._http._transport._pool._keepalive_expiry == 240.0
        await client.close()

    def test_needs_an_account_id_and_an_api_key(self):
        with pytest.raises(ValueError):
            ClefClient(account_id="", api_key="key")
        with pytest.raises(ValueError):
            ClefClient(account_id="acct", api_key="")


class TestClefClassifier:
    @pytest.mark.asyncio
    async def test_yes_no(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["question"] = json.loads(request.content)["questions"]["answer"]
            return _reply({"type": "noul", "noul": 0.95})

        classifier = ClefClassifier(client=_client(handler))
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

        classifier = ClefClassifier(client=_client(handler))
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

        assert seen["question"]["criteria"] == {
            "true": "a recorded greeting",
            "false": "a person talking",
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
                    "choice": "technical",
                    "probabilities": {"billing": 0.16, "technical": 0.83},
                    "confidence": 0.56,
                }
            )

        classifier = ClefClassifier(client=_client(handler))
        options = {"billing": "payments", "technical": "outages", "sales": None}
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
            "criteria": options,
        }
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_too_many_options_is_an_error_before_any_request(self):
        sent = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            return _reply({"type": "choice", "choice": "o0", "confidence": 1.0})

        classifier = ClefClassifier(client=_client(handler))
        options = {f"o{i}": None for i in range(CLEF_MAX_CHOICE_OPTIONS + 1)}
        with pytest.raises(ClassifierError, match="at most 255 options"):
            await classifier.choice(
                "hmm", {"answer": ChoiceQuestion(instructions="?", options=options)}
            )
        assert sent == []
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_a_choice_outside_the_options_is_an_error(self):
        classifier = ClefClassifier(
            client=_client(
                lambda request: _reply({"type": "choice", "choice": "maybe", "confidence": 0.8})
            )
        )
        question = ChoiceQuestion(instructions="?", options={"yes": None, "no": None})
        with pytest.raises(ClassifierError, match="Clef chose 'maybe'"):
            await classifier.choice("hmm", {"answer": question})
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_an_unusable_probability_is_an_error(self):
        classifier = ClefClassifier(
            client=_client(
                lambda request: _reply(
                    {
                        "type": "choice",
                        "choice": "yes",
                        "probabilities": {"yes": "high", "no": 0.1},
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
            seen["question"] = json.loads(request.content)["questions"]["answer"]
            return _reply(
                {
                    "type": "score",
                    "score": 1.98,
                    "legend": {"0": "calm", "1": "impatient", "2": "angry"},
                    "probabilities": {"0": 0.01, "1": 0.01, "2": 0.98},
                    "confidence": 0.96,
                }
            )

        classifier = ClefClassifier(client=_client(handler))
        levels = ["calm", "impatient", "angry"]
        result = (
            await classifier.score(
                "This is the third time!",
                {"answer": ScoreQuestion(instructions="how upset?", levels=levels)},
            )
        )["answer"]

        assert result.score == 1.98
        assert [(l.level, l.probability) for l in result.levels] == [
            ("calm", 0.01),
            ("impatient", 0.01),
            ("angry", 0.98),
        ]
        assert result.confidence == 0.96
        assert seen["question"] == {
            "type": "score",
            "instructions": "how upset?",
            "criteria": levels,
        }
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_a_missing_answer_field_is_an_error(self):
        classifier = ClefClassifier(client=_client(lambda request: _reply({"type": "noul"})))

        with pytest.raises(ClassifierError, match="noul"):
            await classifier.yes_no("a", {"answer": YesNoQuestion(instructions="?")})
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_ask_sends_every_question_in_one_request(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return _success(
                {
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
                }
            )

        classifier = ClefClassifier(client=_client(handler))
        results = await classifier.ask(
            "Hello there!",
            {
                "greeting": YesNoQuestion(instructions="is this a greeting?"),
                "mood": ScoreQuestion(instructions="how upset?", levels=["calm", "upset"]),
            },
        )

        assert len(requests) == 1
        assert set(requests[0]["questions"]) == {"greeting", "mood"}
        assert results["greeting"].probability == 0.9
        assert results["mood"].probability("calm") == 0.8
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_a_missing_answer_is_an_error(self):
        classifier = ClefClassifier(
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

    def test_needs_an_account_and_key_or_a_client(self):
        with pytest.raises(ValueError):
            ClefClassifier()
        with pytest.raises(ValueError):
            ClefClassifier(api_key="key")
        with pytest.raises(ValueError):
            ClefClassifier(account_id="acct")

    def test_a_client_of_its_own_takes_the_model_and_timeout(self):
        classifier = ClefClassifier(
            account_id="acct", api_key="key", model="clef-flash", timeout=1.5
        )
        assert classifier._owns_client
        assert classifier.model == "clef-flash"
        assert classifier.client._http.timeout.read == 1.5

    @pytest.mark.asyncio
    async def test_cleanup_closes_only_an_owned_client(self):
        owned = ClefClassifier(account_id="acct", api_key="key")
        await owned.cleanup()
        assert owned.client._http.is_closed

        shared = ClefClient(account_id="acct", api_key="key")
        classifier = ClefClassifier(client=shared)
        await classifier.cleanup()
        assert not shared._http.is_closed
        await shared.close()

    @pytest.mark.asyncio
    async def test_setup_opens_the_connection(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.path)
            return _schema(request)

        classifier = ClefClassifier(client=_client(handler))
        await classifier.setup(TaskManager())

        assert seen == ["/client/v4/accounts/acct/ai/models/schema"]
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_a_failed_connect_at_setup_is_only_a_warning(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        classifier = ClefClassifier(client=_client(handler))
        await classifier.setup(TaskManager())
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_a_shared_client_connects_once(self):
        connects = []

        def handler(request: httpx.Request) -> httpx.Response:
            connects.append(request.url.path)
            return _schema(request)

        client = _client(handler)
        first, second = ClefClassifier(client=client), ClefClassifier(client=client)
        task_manager = TaskManager()
        await asyncio.gather(first.setup(task_manager), second.setup(task_manager))
        await first.setup(task_manager)

        assert len(connects) == 1
        await client.close()

    @pytest.mark.asyncio
    async def test_every_call_reports_its_time_and_tokens(self):
        client = _client(lambda request: _reply({"type": "noul", "noul": 0.9}))
        classifier = ClefClassifier(client=client, name="voicemail")
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
        assert (processing.processor, processing.model) == ("voicemail", "clef")
        assert isinstance(usage, LLMUsageMetricsData)
        assert (usage.value.prompt_tokens, usage.value.total_tokens) == (12, 12)
        await client.close()


@pytest.mark.skipif(
    not (os.getenv("CLOUDFLARE_ACCOUNT_ID") and os.getenv("CLOUDFLARE_API_KEY")),
    reason="CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_KEY not set",
)
class TestClefLive:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ["clef", "clef-flash"])
    async def test_three_questions(self, model):
        classifier = ClefClassifier(
            account_id=os.environ["CLOUDFLARE_ACCOUNT_ID"],
            api_key=os.environ["CLOUDFLARE_API_KEY"],
            model=model,
        )
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
