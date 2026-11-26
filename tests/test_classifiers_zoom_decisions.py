#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import io
import json
import os
import ssl
from collections.abc import Callable

import httpx
import pytest
from PIL import Image, ImageDraw

from pipecat.classifiers.base_classifier import (
    ChoiceQuestion,
    ClassifierError,
    ClassifierImage,
    ScoreQuestion,
    YesNoQuestion,
)
from pipecat.classifiers.zoom.decisions.classifier import (
    ZOOM_DECISIONS_MAX_CHOICE_OPTIONS,
    ZOOM_DECISIONS_MAX_IMAGES,
    ZOOM_DECISIONS_MAX_QUESTIONS,
    ZOOM_DECISIONS_MAX_SCORE_LEVELS,
    ZoomDecisionsClassifier,
)
from pipecat.classifiers.zoom.decisions.client import ZoomDecisionsClient
from pipecat.metrics.metrics import LLMUsageMetricsData, ProcessingMetricsData
from pipecat.utils.asyncio.task_manager import TaskManager

USAGE = {"input_units": 12, "output_units": 1, "unit_type": "tokens", "details": {}}
PREDICATE = [{"type": "predicate", "name": "answer", "instructions": "?"}]


def _client(handler: Callable[[httpx.Request], httpx.Response], **kwargs) -> ZoomDecisionsClient:
    """A client whose requests ``handler`` answers instead of Zoom."""
    client = ZoomDecisionsClient(api_key="key", **kwargs)
    client._http._transport = httpx.MockTransport(handler)
    return client


def _reply(*answers: dict, usage: object = USAGE) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "answers": list(answers),
            "model": "zoom-decisions",
            "request_id": "ZOAP1",
            "usage": usage,
        },
    )


def _predicate(probability: float, name: str = "answer") -> dict:
    return {"type": "predicate", "name": name, "probability": probability}


def _failure(status: int, message: str, reason: str = "INVALID_INPUT", **headers) -> httpx.Response:
    return httpx.Response(
        status,
        json={"code": status, "reason": reason, "message": message, "metadata": {}},
        headers=headers,
    )


def _no_sleep(monkeypatch) -> list[float]:
    """Skip the client's waits between retries, and return how long each would have been."""
    waits: list[float] = []

    async def no_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr("pipecat.classifiers.zoom.decisions.client.asyncio.sleep", no_sleep)
    return waits


def _circle(color: str) -> ClassifierImage:
    """A camera-sized JPEG with a circle in the middle."""
    image = Image.new("RGB", (640, 480), "white")
    ImageDraw.Draw(image).ellipse((220, 140, 420, 340), fill=color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return ClassifierImage(data=buffer.getvalue(), content_type="image/jpeg")


class TestZoomDecisionsClient:
    @pytest.mark.asyncio
    async def test_sends_questions_to_the_decide_endpoint_with_auth_and_tracking(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["headers"] = request.headers
            seen["body"] = json.loads(request.content)
            return _reply(_predicate(0.9))

        client = _client(handler)
        answers, usage = await client.ask("hello", PREDICATE)

        assert answers["answer"] == _predicate(0.9)
        assert (usage.input_tokens, usage.output_tokens) == (12, 1)
        assert seen["url"] == "https://api.zoom.us/v2/aiservices/decisions/decide"
        assert seen["headers"]["Authorization"] == "Bearer key"
        assert seen["headers"]["x-zm-trackingid"].startswith("zoom_ai_community_pipecat_")
        assert seen["body"] == {"model": "zoom-decisions", "input": "hello", "questions": PREDICATE}
        await client.close()

    @pytest.mark.asyncio
    async def test_the_url_model_and_options_shape_the_request(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            return _reply(_predicate(0.9))

        client = _client(handler, url="https://example.com/decide", model="zoom-decisions-pro")
        await client.ask("hello", PREDICATE, enable_image_input=True)

        assert seen["url"] == "https://example.com/decide"
        assert seen["body"]["model"] == "zoom-decisions-pro"
        assert seen["body"]["enable_image_input"] is True
        await client.close()

    @pytest.mark.asyncio
    async def test_answers_are_matched_to_questions_by_name(self):
        client = _client(lambda request: _reply(_predicate(0.2, "b"), _predicate(0.8, "a")))
        answers, _ = await client.ask(
            "x",
            [
                {"type": "predicate", "name": "a", "instructions": "?"},
                {"type": "predicate", "name": "b", "instructions": "?"},
            ],
        )

        assert list(answers) == ["a", "b"]
        assert answers["a"]["probability"] == 0.8
        assert answers["b"]["probability"] == 0.2
        await client.close()

    @pytest.mark.asyncio
    async def test_counts_tokens_over_requests_cached_ones_too(self):
        usage = {**USAGE, "details": {"cached_input_tokens": 8}}
        client = _client(lambda request: _reply(_predicate(0.5), usage=usage))
        await client.ask("a", PREDICATE)
        await client.ask("b", PREDICATE)

        assert client.usage.input_tokens == 24
        assert client.usage.output_tokens == 2
        assert client.usage.cached_input_tokens == 16
        await client.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("usage", ["none", {"input_units": "12", "details": []}])
    async def test_usage_that_cant_be_read_counts_as_none(self, usage):
        client = _client(lambda request: _reply(_predicate(0.5), usage=usage))
        _, request_usage = await client.ask("a", PREDICATE)

        assert request_usage.input_tokens == 0
        assert client.usage.cached_input_tokens == 0
        await client.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("busy", [429, 503])
    async def test_retries_when_busy(self, monkeypatch, busy):
        waits = _no_sleep(monkeypatch)
        statuses = iter([busy, busy])

        def handler(request: httpx.Request) -> httpx.Response:
            status = next(statuses, None)
            if status is not None:
                return _failure(status, "busy", "SERVICE_UNAVAILABLE")
            return _reply(_predicate(0.7))

        client = _client(handler)
        answers, _ = await client.ask("a", PREDICATE)

        assert answers["answer"]["probability"] == 0.7
        assert waits == [0.25, 0.5]
        await client.close()

    @pytest.mark.asyncio
    async def test_waits_as_long_as_retry_after_asks(self, monkeypatch):
        waits = _no_sleep(monkeypatch)
        statuses = iter([503])

        def handler(request: httpx.Request) -> httpx.Response:
            if next(statuses, None):
                return _failure(503, "busy", "SERVICE_UNAVAILABLE", **{"Retry-After": "1.5"})
            return _reply(_predicate(0.7))

        client = _client(handler)
        await client.ask("a", PREDICATE)

        assert waits == [1.5]
        await client.close()

    @pytest.mark.asyncio
    async def test_gives_up_at_once_when_retry_after_is_past_the_timeout(self, monkeypatch):
        waits = _no_sleep(monkeypatch)
        client = _client(
            lambda request: _failure(
                503, "temporarily unavailable", "SERVICE_UNAVAILABLE", **{"Retry-After": "10"}
            ),
            timeout=2.0,
        )

        with pytest.raises(ClassifierError, match=r"busy \(HTTP 503: temporarily unavailable"):
            await client.ask("a", PREDICATE)
        assert waits == []
        await client.close()

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries_with_zooms_reason(self, monkeypatch):
        _no_sleep(monkeypatch)
        client = _client(lambda request: _failure(429, "Too many requests"), max_retries=2)

        with pytest.raises(
            ClassifierError, match=r"busy \(HTTP 429: Too many requests .*3 attempts"
        ):
            await client.ask("a", PREDICATE)
        await client.close()

    @pytest.mark.asyncio
    async def test_a_rejected_request_carries_zooms_reason_and_the_tracking_id(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["tracking_id"] = request.headers["x-zm-trackingid"]
            return _failure(400, "questions[0].instructions is required")

        client = _client(handler)

        with pytest.raises(ClassifierError) as error:
            await client.ask("a", PREDICATE)
        assert "rejected the request: HTTP 400: questions[0].instructions is required" in str(
            error.value
        )
        assert f"tracking_id={seen['tracking_id']}" in str(error.value)
        await client.close()

    @pytest.mark.asyncio
    async def test_a_rejected_request_without_a_reason_has_only_the_status(self):
        client = _client(lambda request: httpx.Response(502, text="<html>bad gateway</html>"))

        with pytest.raises(ClassifierError, match=r"rejected the request: HTTP 502 \(tracking_id="):
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
        client = _client(lambda request: httpx.Response(200, json={"model": "zoom-decisions"}))

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
    async def test_no_answer_in_time_is_an_error(self):
        async def slow(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(1)
            return _reply(_predicate(0.5))

        client = ZoomDecisionsClient(api_key="key", timeout=0.05)
        client._http._transport = httpx.MockTransport(slow)

        with pytest.raises(ClassifierError, match="did not answer within 0.05s"):
            await client.ask("a", PREDICATE)
        await client.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error", [httpx.ConnectError("down"), ssl.SSLError("bad record mac")])
    async def test_unreachable_is_an_error(self, error):
        def handler(request: httpx.Request) -> httpx.Response:
            raise error

        client = _client(handler)

        with pytest.raises(ClassifierError, match="request failed"):
            await client.ask("a", PREDICATE)
        with pytest.raises(ClassifierError, match="could not connect"):
            await client.connect()
        await client.close()

    @pytest.mark.asyncio
    async def test_connect_opens_the_connection_once_without_asking(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((request.method, request.url.path))
            return httpx.Response(200)

        client = _client(handler)
        await client.connect()
        await client.connect()

        assert seen == [("OPTIONS", "/v2/aiservices/decisions/decide")]
        await client.close()

    @pytest.mark.asyncio
    async def test_keeps_idle_connections_open_between_questions(self):
        client = ZoomDecisionsClient(api_key="key")
        assert client._http._transport._pool._keepalive_expiry == 240.0
        await client.close()

    def test_needs_an_api_key(self):
        with pytest.raises(ValueError):
            ZoomDecisionsClient(api_key="")


def _questions(request: httpx.Request) -> list[dict]:
    return json.loads(request.content)["questions"]


class TestZoomDecisionsClassifier:
    @pytest.mark.asyncio
    async def test_yes_no(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["questions"] = _questions(request)
            return _reply(_predicate(0.95))

        classifier = ZoomDecisionsClassifier(client=_client(handler))
        result = (
            await classifier.yes_no(
                "Please leave a message",
                {"answer": YesNoQuestion(instructions="is this a voicemail greeting?")},
            )
        )["answer"]

        assert result.probability == 0.95
        assert seen["questions"] == [
            {
                "type": "predicate",
                "name": "answer",
                "instructions": "is this a voicemail greeting?",
            }
        ]
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_structured_state_is_sent_as_json(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["input"] = json.loads(request.content)["input"]
            return _reply(_predicate(0.5))

        classifier = ZoomDecisionsClassifier(client=_client(handler))
        state = [{"role": "user", "content": "hi"}]
        await classifier.yes_no(state, {"answer": YesNoQuestion(instructions="a greeting?")})

        assert json.loads(seen["input"]) == state
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_choice(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["questions"] = _questions(request)
            return _reply(
                {
                    "type": "choice",
                    "name": "answer",
                    "choice": "technical",
                    "probabilities": [
                        {"value": "billing", "probability": 0.16},
                        {"value": "technical", "probability": 0.83},
                    ],
                }
            )

        classifier = ZoomDecisionsClassifier(client=_client(handler))
        options = {"billing": "payments", "technical": None, "sales": None}
        result = (
            await classifier.choice(
                "Checkout is down",
                {"answer": ChoiceQuestion(instructions="which team?", options=options)},
            )
        )["answer"]

        assert result.choice == "technical"
        assert result.probabilities == {"billing": 0.16, "technical": 0.83, "sales": 0.0}
        assert result.confidence == 0.83
        assert seen["questions"][0]["choices"] == [
            {"value": "billing", "description": "payments"},
            {"value": "technical"},
            {"value": "sales"},
        ]
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_score(self):
        classifier = ZoomDecisionsClassifier(
            client=_client(
                lambda request: _reply(
                    {
                        "type": "score",
                        "name": "answer",
                        "score": 1.9,
                        "probabilities": [
                            {"value": 0, "probability": 0.05},
                            {"value": 1, "probability": 0.0},
                            {"value": 2, "probability": 0.95},
                        ],
                    }
                )
            )
        )
        levels = ["calm", "impatient", "angry"]
        result = (
            await classifier.score(
                "This is the third time!",
                {"answer": ScoreQuestion(instructions="how upset?", levels=levels)},
            )
        )["answer"]

        assert result.score == 1.9
        assert result.probability("angry") == 0.95
        assert result.confidence == 0.95
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_images_go_in_a_user_message_and_turn_on_image_input(self):
        bodies = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return _reply(_predicate(0.99))

        classifier = ZoomDecisionsClassifier(client=_client(handler))
        png = ClassifierImage(data=b"png bytes", content_type="image/png")
        question = {"answer": YesNoQuestion(instructions="is there a red circle?")}
        assert classifier.supports_images

        await classifier.yes_no("a camera frame", question, inputs=[png])
        await classifier.yes_no("no frame", question)

        assert bodies[0]["enable_image_input"] is True
        assert bodies[0]["input"] == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,"
                        + base64.b64encode(b"png bytes").decode(),
                    },
                    {"type": "input_text", "text": "a camera frame"},
                ],
            }
        ]
        assert "enable_image_input" not in bodies[1]
        assert bodies[1]["input"] == "no frame"
        await classifier.client.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "questions,images,match",
        [
            (
                {f"q{i}": YesNoQuestion(instructions="?") for i in range(11)},
                [],
                f"at most {ZOOM_DECISIONS_MAX_QUESTIONS} questions",
            ),
            (
                {
                    "answer": ChoiceQuestion(
                        instructions="?",
                        options={
                            f"o{i}": None for i in range(ZOOM_DECISIONS_MAX_CHOICE_OPTIONS + 1)
                        },
                    )
                },
                [],
                "at most 255 options, got 256 for 'answer'",
            ),
            (
                {
                    "answer": ScoreQuestion(
                        instructions="?",
                        levels=[f"l{i}" for i in range(ZOOM_DECISIONS_MAX_SCORE_LEVELS + 1)],
                    )
                },
                [],
                "at most 10 levels, got 11 for 'answer'",
            ),
            (
                {"answer": YesNoQuestion(instructions="?")},
                [ClassifierImage(data=b"x", content_type="image/png")]
                * (ZOOM_DECISIONS_MAX_IMAGES + 1),
                "at most 4 images",
            ),
        ],
    )
    async def test_a_request_past_zooms_limits_is_an_error_before_any_request(
        self, questions, images, match
    ):
        sent = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            return _reply(_predicate(1.0))

        classifier = ZoomDecisionsClassifier(client=_client(handler))
        with pytest.raises(ClassifierError, match=match):
            await classifier.ask("hmm", questions, inputs=images)
        assert sent == []
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_a_refusal_is_an_error_naming_the_question(self):
        classifier = ZoomDecisionsClassifier(
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
                        {"value": 0, "probability": 0.8},
                        {"value": 1, "probability": 0.2},
                    ],
                },
            )

        classifier = ZoomDecisionsClassifier(client=_client(handler))
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

    def test_needs_an_api_key_or_a_client(self):
        with pytest.raises(ValueError):
            ZoomDecisionsClassifier()

    def test_a_client_of_its_own_takes_the_model_url_and_timeout(self):
        classifier = ZoomDecisionsClassifier(
            api_key="key", model="zoom-decisions-pro", url="https://example.com/decide", timeout=1.5
        )
        assert classifier._owns_client
        assert classifier.model == "zoom-decisions-pro"
        assert classifier.client._url == "https://example.com/decide"
        assert classifier.client._timeout == 1.5

    @pytest.mark.asyncio
    async def test_cleanup_closes_only_an_owned_client(self):
        owned = ZoomDecisionsClassifier(api_key="key")
        await owned.cleanup()
        assert owned.client._http.is_closed

        shared = ZoomDecisionsClient(api_key="key")
        classifier = ZoomDecisionsClassifier(client=shared)
        await classifier.cleanup()
        assert not shared._http.is_closed
        await shared.close()

    @pytest.mark.asyncio
    async def test_setup_opens_the_connection_and_a_shared_client_connects_once(self):
        connects = []

        def handler(request: httpx.Request) -> httpx.Response:
            connects.append(request.method)
            return httpx.Response(200)

        client = _client(handler)
        first = ZoomDecisionsClassifier(client=client)
        second = ZoomDecisionsClassifier(client=client)
        task_manager = TaskManager()
        await asyncio.gather(first.setup(task_manager), second.setup(task_manager))

        assert connects == ["OPTIONS"]
        await client.close()

    @pytest.mark.asyncio
    async def test_a_failed_connect_at_setup_is_only_a_warning(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        classifier = ZoomDecisionsClassifier(client=_client(handler))
        await classifier.setup(TaskManager())
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_every_call_reports_its_time_and_tokens(self):
        usage = {**USAGE, "details": {"cached_input_tokens": 8}}
        client = _client(lambda request: _reply(_predicate(0.9), usage=usage))
        classifier = ZoomDecisionsClassifier(client=client, name="voicemail")
        reported = asyncio.Event()
        seen: list = []

        @classifier.event_handler("on_metrics")
        async def on_metrics(classifier, data):
            seen.extend(data)
            reported.set()

        await classifier.yes_no("hello", {"answer": YesNoQuestion(instructions="a greeting?")})
        await asyncio.wait_for(reported.wait(), 1)

        processing, usage_metrics = seen
        assert isinstance(processing, ProcessingMetricsData)
        assert (processing.processor, processing.model) == ("voicemail", "zoom-decisions")
        assert isinstance(usage_metrics, LLMUsageMetricsData)
        value = usage_metrics.value
        assert (value.prompt_tokens, value.completion_tokens, value.total_tokens) == (12, 1, 13)
        assert value.cache_read_input_tokens == 8
        await client.close()


@pytest.mark.skipif(not os.getenv("ZOOM_AI_API_KEY"), reason="ZOOM_AI_API_KEY not set")
class TestZoomDecisionsLive:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ["zoom-decisions", "zoom-decisions-pro"])
    async def test_three_questions(self, model):
        classifier = ZoomDecisionsClassifier(api_key=os.environ["ZOOM_AI_API_KEY"], model=model)
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

    @pytest.mark.asyncio
    async def test_a_question_about_an_image(self):
        classifier = ZoomDecisionsClassifier(api_key=os.environ["ZOOM_AI_API_KEY"])
        question = {"red": YesNoQuestion(instructions="Is there a red circle in the image?")}
        try:
            red = await classifier.yes_no("A camera frame.", question, inputs=[_circle("red")])
            blue = await classifier.yes_no("A camera frame.", question, inputs=[_circle("blue")])
            assert red["red"].is_yes
            assert not blue["red"].is_yes
        finally:
            await classifier.cleanup()
