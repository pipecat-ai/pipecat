#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the classifier backed by Hume's Audio Decisions API."""

import io
import json
import wave
from collections.abc import Callable
from email.parser import BytesParser
from email.policy import default

import httpx
import pytest

from pipecat.classifiers.base_classifier import (
    ChoiceQuestion,
    ChoiceResult,
    ClassifierAudio,
    ClassifierError,
    ScoreLevel,
    ScoreQuestion,
    ScoreResult,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.classifiers.hume.audio_decisions.classifier import HumeAudioClassifier
from pipecat.classifiers.hume.audio_decisions.client import HumeAudioDecisionsClient
from pipecat.metrics.metrics import LLMUsageMetricsData
from pipecat.utils.asyncio.task_manager import TaskManager

TONE = ChoiceQuestion(
    instructions="How does the speaker sound?",
    options={"calm": None, "frustrated": "Annoyed, short or raised voice."},
)
HESITANT = YesNoQuestion(
    instructions="Does the speaker sound unsure?", yes="They pause or trail off."
)
ENERGY = ScoreQuestion(
    instructions="How energetic is the speaker?", levels=["low", "medium", "high"]
)

ANSWERS = {
    "tone": {
        "type": "choice",
        "choice": "frustrated",
        "confidence": 0.41,
        "probabilities": {"calm": 0.3, "frustrated": 0.7},
    },
    "hesitant": {"type": "noul", "noul": 0.62},
    "energy": {
        "type": "score",
        "score": 1.4,
        "confidence": 0.5,
        "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
        "legend": {"0": "low", "1": "medium", "2": "high"},
    },
}


def _client(
    handler: Callable[[httpx.Request], httpx.Response], **kwargs
) -> HumeAudioDecisionsClient:
    """A client whose requests ``handler`` answers instead of Hume."""
    client = HumeAudioDecisionsClient(api_key="key", **kwargs)
    client._http._transport = httpx.MockTransport(handler)
    return client


def _reply(answers: dict, input_tokens: int = 1200) -> httpx.Response:
    return httpx.Response(
        200, json={"run_id": "run", "answers": answers, "received": {"input_tokens": input_tokens}}
    )


def _parts(request: httpx.Request) -> list[tuple[str, bytes | str]]:
    """The multipart parts of a request, in order, as (name, content)."""
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.read()
    )
    return [
        (part.get_param("name", header="content-disposition"), part.get_content())
        for part in message.iter_parts()
    ]


def _request_json(request: httpx.Request) -> dict:
    return json.loads(dict(_parts(request))["request"])


def _clip(secs: float = 6.0, sample_rate: int = 16000) -> ClassifierAudio:
    return ClassifierAudio(audio=bytes(int(secs * sample_rate) * 2), sample_rate=sample_rate)


@pytest.mark.asyncio
async def test_the_request_goes_before_the_file_with_the_api_key():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _reply({"hesitant": ANSWERS["hesitant"]})

    classifier = HumeAudioClassifier(client=_client(handler))
    await classifier.ask("A caller.", {"hesitant": HESITANT}, inputs=[_clip()])

    [request] = seen
    assert request.url.path == "/v0/decisions/file"
    assert request.headers["X-Hume-Api-Key"] == "key"
    assert [name for name, _ in _parts(request)] == ["request", "file"]


@pytest.mark.asyncio
async def test_questions_are_sent_in_humes_own_format():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(_request_json(request))
        return _reply(ANSWERS)

    classifier = HumeAudioClassifier(client=_client(handler))
    await classifier.ask(
        "A caller.", {"tone": TONE, "hesitant": HESITANT, "energy": ENERGY}, inputs=[_clip()]
    )

    assert seen[0]["questions"] == {
        "tone": {
            "type": "choice",
            "instructions": "How does the speaker sound?",
            "criteria": [
                {"name": "calm"},
                {"name": "frustrated", "description": "Annoyed, short or raised voice."},
            ],
        },
        "hesitant": {
            "type": "noul",
            "instructions": "Does the speaker sound unsure?",
            "criteria": {"true": "They pause or trail off."},
        },
        "energy": {
            "type": "score",
            "instructions": "How energetic is the speaker?",
            "criteria": ["low", "medium", "high"],
        },
    }


@pytest.mark.asyncio
async def test_a_yes_no_question_without_criteria_sends_none():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(_request_json(request))
        return _reply({"hesitant": ANSWERS["hesitant"]})

    classifier = HumeAudioClassifier(client=_client(handler))
    await classifier.ask(
        "A caller.",
        {"hesitant": YesNoQuestion(instructions={"ask": "Unsure?", "about": "the speaker"})},
        inputs=[_clip()],
    )

    assert seen[0]["questions"]["hesitant"] == {
        "type": "noul",
        "instructions": json.dumps({"ask": "Unsure?", "about": "the speaker"}),
    }


@pytest.mark.asyncio
async def test_answers_become_typed_results():
    classifier = HumeAudioClassifier(client=_client(lambda request: _reply(ANSWERS)))

    results = await classifier.ask(
        "A caller.", {"tone": TONE, "hesitant": HESITANT, "energy": ENERGY}, inputs=[_clip()]
    )

    assert results == {
        "tone": ChoiceResult(
            choice="frustrated", probabilities={"calm": 0.3, "frustrated": 0.7}, confidence=0.41
        ),
        "hesitant": YesNoResult(probability=0.62),
        "energy": ScoreResult(
            score=1.4,
            levels=[
                ScoreLevel(level="low", probability=0.1),
                ScoreLevel(level="medium", probability=0.4),
                ScoreLevel(level="high", probability=0.5),
            ],
            confidence=0.5,
        ),
    }


@pytest.mark.asyncio
async def test_the_clip_is_sent_as_a_16khz_mono_wav():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(_parts(request))["file"])
        return _reply({"hesitant": ANSWERS["hesitant"]})

    classifier = HumeAudioClassifier(client=_client(handler))
    await classifier.ask("A caller.", {"hesitant": HESITANT}, inputs=[_clip(6, sample_rate=8000)])

    with wave.open(io.BytesIO(seen[0]), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        assert wav.getnframes() / 16000 == pytest.approx(6, abs=0.05)


@pytest.mark.asyncio
async def test_a_clip_shorter_than_five_seconds_is_refused_without_a_request():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _reply({"hesitant": ANSWERS["hesitant"]})

    classifier = HumeAudioClassifier(client=_client(handler))

    with pytest.raises(ClassifierError):
        await classifier.ask("A caller.", {"hesitant": HESITANT}, inputs=[_clip(4.5)])

    assert seen == []


@pytest.mark.asyncio
async def test_questions_without_a_clip_are_refused_without_a_request():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _reply({"hesitant": ANSWERS["hesitant"]})

    classifier = HumeAudioClassifier(client=_client(handler))

    assert classifier.supports_audio
    with pytest.raises(ClassifierError, match="clip of audio"):
        await classifier.ask("A caller.", {"hesitant": HESITANT})

    assert seen == []


@pytest.mark.asyncio
async def test_two_clips_are_refused_without_a_request():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _reply({"hesitant": ANSWERS["hesitant"]})

    classifier = HumeAudioClassifier(client=_client(handler))

    with pytest.raises(ClassifierError, match="one clip of audio, got 2"):
        await classifier.ask("A caller.", {"hesitant": HESITANT}, inputs=[_clip(), _clip()])

    assert seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state, context",
    [
        ("A caller.", "A caller."),
        ({"assistant": "What's your order number?"}, '{"assistant": "What\'s your order number?"}'),
        ("x" * 500 + "y" * 1000, "y" * 1000),
        ({}, None),
    ],
)
async def test_the_state_is_sent_as_the_context(state, context):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(_request_json(request))
        return _reply({"hesitant": ANSWERS["hesitant"]})

    classifier = HumeAudioClassifier(client=_client(handler))
    await classifier.ask(state, {"hesitant": HESITANT}, inputs=[_clip()])

    assert seen[0].get("context") == context


@pytest.mark.asyncio
async def test_the_billed_input_tokens_are_counted_and_reported():
    client = _client(lambda request: _reply({"hesitant": ANSWERS["hesitant"]}, input_tokens=900))
    classifier = HumeAudioClassifier(client=client)
    reported = []

    @classifier.event_handler("on_metrics")
    async def on_metrics(classifier, data):
        reported.extend(data)

    for _ in range(2):
        await classifier.ask("A caller.", {"hesitant": HESITANT}, inputs=[_clip()])
    await classifier.cleanup()

    assert client.usage.input_tokens == 1800
    usage = [d.value for d in reported if isinstance(d, LLMUsageMetricsData)]
    assert [u.prompt_tokens for u in usage] == [900, 900]


@pytest.mark.asyncio
async def test_a_busy_hume_is_retried_after_its_retry_after(monkeypatch):
    waits = []
    replies = [httpx.Response(503, headers={"Retry-After": "0.5"}), _reply(ANSWERS)]

    async def no_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr("pipecat.classifiers.hume.audio_decisions.client.asyncio.sleep", no_sleep)
    classifier = HumeAudioClassifier(client=_client(lambda request: replies.pop(0)))

    results = await classifier.ask("A caller.", {"hesitant": HESITANT}, inputs=[_clip()])

    assert results["hesitant"] == YesNoResult(probability=0.62)
    assert waits == [0.5]


@pytest.mark.asyncio
async def test_a_long_retry_after_is_reported_without_waiting(monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(429, headers={"Retry-After": "30"})

    classifier = HumeAudioClassifier(client=_client(handler))

    with pytest.raises(ClassifierError, match="retry after 30s"):
        await classifier.ask("A caller.", {"hesitant": HESITANT}, inputs=[_clip()])

    assert len(seen) == 1


@pytest.mark.asyncio
async def test_a_rejection_reports_humes_code_and_message():
    error = {
        "type": "error",
        "code": "question_invalid",
        "message": "Option names must be unique.",
        "param": "questions.tone",
        "request_id": "req",
    }
    classifier = HumeAudioClassifier(
        client=_client(lambda request: httpx.Response(400, json=error))
    )

    with pytest.raises(ClassifierError, match="question_invalid Option names must be unique"):
        await classifier.ask("A caller.", {"tone": TONE}, inputs=[_clip()])


@pytest.mark.asyncio
async def test_a_choice_that_isnt_an_option_is_an_error():
    answer = {**ANSWERS["tone"], "choice": "elated"}
    classifier = HumeAudioClassifier(client=_client(lambda request: _reply({"tone": answer})))

    with pytest.raises(ClassifierError):
        await classifier.ask("A caller.", {"tone": TONE}, inputs=[_clip()])


@pytest.mark.asyncio
async def test_a_question_left_unanswered_is_an_error():
    classifier = HumeAudioClassifier(
        client=_client(lambda request: _reply({"tone": ANSWERS["tone"]}))
    )

    with pytest.raises(ClassifierError, match="no answer for hesitant"):
        await classifier.ask("A caller.", {"tone": TONE, "hesitant": HESITANT}, inputs=[_clip()])


@pytest.mark.asyncio
async def test_setup_opens_the_connection_once_for_a_shared_client():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, dict(request.url.params)))
        return httpx.Response(200, json={"runs": []})

    client = _client(handler)
    for classifier in (
        HumeAudioClassifier(client=client),
        HumeAudioClassifier(client=client),
    ):
        await classifier.setup(TaskManager())

    assert seen == [("GET", "/v0/decisions/run", {"page_size": "1"})]


@pytest.mark.asyncio
async def test_a_connection_that_cant_be_opened_at_setup_is_only_a_warning():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(503)
        return _reply({"hesitant": ANSWERS["hesitant"]})

    classifier = HumeAudioClassifier(client=_client(handler))
    await classifier.setup(TaskManager())

    results = await classifier.ask("A caller.", {"hesitant": HESITANT}, inputs=[_clip()])

    assert results["hesitant"] == YesNoResult(probability=0.62)


@pytest.mark.asyncio
async def test_a_client_of_its_own_is_closed_and_a_shared_one_is_not():
    own = HumeAudioClassifier(api_key="key")
    shared_client = HumeAudioDecisionsClient(api_key="key")
    shared = HumeAudioClassifier(client=shared_client)

    await own.cleanup()
    await shared.cleanup()

    assert own.client._http.is_closed
    assert not shared_client._http.is_closed
    await shared_client.close()
