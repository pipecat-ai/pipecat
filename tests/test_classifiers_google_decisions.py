#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import io
import json
import math
import os
import ssl
import struct
import wave
from collections.abc import Callable

import httpx
import pytest
from PIL import Image, ImageDraw, ImageFont

from pipecat.classifiers.base_classifier import (
    ChoiceQuestion,
    ClassifierAudio,
    ClassifierError,
    ClassifierFile,
    ClassifierImage,
    ScoreQuestion,
    YesNoQuestion,
)
from pipecat.classifiers.google.decisions.classifier import (
    GOOGLE_DECISIONS_FILE_TYPES,
    GOOGLE_DECISIONS_MAX_CHOICE_OPTIONS,
    GoogleDecisionsClassifier,
)
from pipecat.classifiers.google.decisions.client import GoogleDecisionsClient
from pipecat.metrics.metrics import LLMUsageMetricsData, ProcessingMetricsData
from pipecat.utils.asyncio.task_manager import TaskManager

USAGE = {"total_input_tokens": 12, "total_output_tokens": 0, "total_tokens": 12}
NOUL = {"answer": {"type": "noul", "instructions": "?"}}
#: A short public video of a fire, from Google's own Decisions examples.
FIRE_VIDEO = "https://storage.googleapis.com/generativeai-downloads/videos/ForBiggerBlazes.mp4"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _client(handler: Callable, **kwargs) -> GoogleDecisionsClient:
    """A client whose requests ``handler`` answers instead of Google."""
    client = GoogleDecisionsClient(api_key="key", **kwargs)
    client._http._transport = httpx.MockTransport(handler)
    return client


def _reply(answer: dict) -> httpx.Response:
    """A reply with one answer, to the question named "answer"."""
    return _replies({"answer": answer})


def _replies(answers: dict[str, dict]) -> httpx.Response:
    """A reply with the answers by question name."""
    return httpx.Response(200, json={"answers": answers, "usage": USAGE})


def _noul(probability: float) -> dict:
    return {"type": "noul", "noul": probability, "confidence": max(probability, 1 - probability)}


def _failure(status: int, message: str) -> httpx.Response:
    return httpx.Response(
        status, json={"error": {"code": status, "message": message, "status": "INVALID_ARGUMENT"}}
    )


def _model(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"name": "models/gemini-decision-1"})


def _question(request: httpx.Request) -> dict:
    """The one question a request sent."""
    (question,) = json.loads(request.content)["questions"].values()
    return question


def _circle(color: str) -> ClassifierImage:
    """A camera-sized JPEG with a circle in the middle."""
    image = Image.new("RGB", (640, 480), "white")
    ImageDraw.Draw(image).ellipse((220, 140, 420, 340), fill=color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return ClassifierImage(data=buffer.getvalue(), content_type="image/jpeg")


def _tone(secs: float, sample_rate: int = 16000) -> ClassifierAudio:
    """A 440 Hz tone as 16-bit mono PCM."""
    samples = int(secs * sample_rate)
    pcm = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / sample_rate)))
        for i in range(samples)
    )
    return ClassifierAudio(audio=pcm, sample_rate=sample_rate)


def _invoice() -> ClassifierFile:
    """A one-page PDF invoice whose total is 240 USD."""
    page = Image.new("RGB", (1240, 1754), "white")
    text = "INVOICE 8821\n\nHeadphones\nTotal: 240 USD"
    ImageDraw.Draw(page).text((120, 200), text, fill="black", font=ImageFont.load_default(64))
    buffer = io.BytesIO()
    page.save(buffer, format="PDF")
    return ClassifierFile(content_type="application/pdf", data=buffer.getvalue())


class TestGoogleDecisionsClient:
    @pytest.mark.asyncio
    async def test_sends_questions_to_the_decisions_endpoint_with_the_key(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["key"] = request.headers["x-goog-api-key"]
            seen["body"] = json.loads(request.content)
            return _reply(_noul(0.9))

        client = _client(handler)
        answers, usage = await client.ask(
            "hello", {"answer": {"type": "noul", "instructions": "a greeting?"}}
        )

        assert answers["answer"].noul == 0.9
        assert (usage.input_tokens, usage.output_tokens) == (12, 0)
        assert seen["url"] == "https://generativelanguage.googleapis.com/v1beta/decisions"
        assert seen["key"] == "key"
        assert seen["body"] == {
            "model": "gemini-decision-1",
            "input": "hello",
            "questions": {"answer": {"type": "noul", "instructions": "a greeting?"}},
        }
        await client.close()

    @pytest.mark.asyncio
    async def test_the_model_picks_where_questions_go(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["model"] = json.loads(request.content)["model"]
            return _reply(_noul(0.9))

        client = _client(handler, model="gemini-decision-next")
        await client.ask("a", NOUL)

        assert seen == {"model": "gemini-decision-next"}
        await client.close()

    @pytest.mark.asyncio
    async def test_answers_are_matched_to_questions_by_name(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["names"] = list(json.loads(request.content)["questions"])
            return _replies({"b": _noul(0.2), "a": _noul(0.8)})

        client = _client(handler)
        answers, _ = await client.ask(
            "x",
            {
                "a": {"type": "noul", "instructions": "a?"},
                "b": {"type": "noul", "instructions": "b?"},
            },
        )

        assert seen["names"] == ["a", "b"]
        assert answers["a"].noul == 0.8
        assert answers["b"].noul == 0.2
        await client.close()

    @pytest.mark.asyncio
    async def test_counts_tokens_over_requests(self):
        client = _client(lambda request: _reply(_noul(0.5)))
        await client.ask("a", NOUL)
        await client.ask("b", NOUL)

        assert client.usage.input_tokens == 24
        await client.close()

    @pytest.mark.asyncio
    async def test_retries_when_busy(self):
        statuses = iter([429])
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            status = next(statuses, None)
            if status is not None:
                return _failure(status, "Resource has been exhausted")
            return _reply(_noul(0.7))

        client = _client(handler)
        answers, _ = await client.ask("a", NOUL)

        assert answers["answer"].noul == 0.7
        assert len(requests) == 2
        await client.close()

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries_with_googles_reason(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return _failure(429, "Resource has been exhausted")

        client = _client(handler, max_retries=1)

        with pytest.raises(ClassifierError, match="HTTP 429: Resource has been exhausted"):
            await client.ask("a", NOUL)
        assert len(requests) == 2
        await client.close()

    @pytest.mark.asyncio
    async def test_a_rejected_request_carries_googles_reason(self):
        client = _client(lambda request: _failure(400, "Question key 'a b' is invalid"))

        with pytest.raises(
            ClassifierError, match="request failed: HTTP 400: Question key 'a b' is invalid"
        ):
            await client.ask("a", NOUL)
        await client.close()

    @pytest.mark.asyncio
    async def test_googles_reason_is_stripped(self):
        client = _client(lambda request: _failure(400, " question IDs must match [A-Za-z0-9_-]+\n"))

        with pytest.raises(ClassifierError) as error:
            await client.ask("a", NOUL)
        assert str(error.value) == (
            "Google Decisions request failed: HTTP 400: question IDs must match [A-Za-z0-9_-]+"
        )
        await client.close()

    @pytest.mark.asyncio
    async def test_a_rejected_request_without_a_reason_has_only_the_status(self):
        client = _client(lambda request: httpx.Response(404, text="<html>not found</html>"))

        with pytest.raises(ClassifierError, match="request failed: HTTP 404$"):
            await client.ask("a", NOUL)
        await client.close()

    @pytest.mark.asyncio
    async def test_a_blocked_input_is_an_error_with_googles_reason(self):
        client = _client(
            lambda request: httpx.Response(
                200,
                json={
                    "prompt_feedback": {
                        "block_reason": "safety",
                        "block_reason_message": "unsafe input",
                    },
                    "usage": {"total_input_tokens": 3},
                },
            )
        )

        with pytest.raises(ClassifierError, match="blocked the input: safety: unsafe input"):
            await client.ask("a", NOUL)
        assert client.usage.input_tokens == 3
        await client.close()

    @pytest.mark.asyncio
    async def test_a_reply_without_answers_is_an_error(self):
        client = _client(lambda request: httpx.Response(200, json={"usage": USAGE}))

        with pytest.raises(ClassifierError, match="no answers"):
            await client.ask("a", NOUL)
        await client.close()

    @pytest.mark.asyncio
    async def test_a_question_left_unanswered_is_an_error_naming_it(self):
        client = _client(lambda request: _reply(_noul(0.5)))

        with pytest.raises(ClassifierError, match="no answer for other"):
            await client.ask(
                "a",
                {
                    "answer": {"type": "noul", "instructions": "?"},
                    "other": {"type": "noul", "instructions": "?"},
                },
            )
        await client.close()

    @pytest.mark.asyncio
    async def test_a_reply_without_usage_counts_no_tokens(self):
        client = _client(
            lambda request: httpx.Response(200, json={"answers": {"answer": _noul(0.5)}})
        )
        await client.ask("a", NOUL)

        assert client.usage.input_tokens == 0
        await client.close()

    @pytest.mark.asyncio
    async def test_unreachable_is_an_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        client = _client(handler, max_retries=0)

        with pytest.raises(ClassifierError, match="request failed"):
            await client.ask("a", NOUL)
        await client.close()

    @pytest.mark.asyncio
    async def test_a_broken_tls_connection_is_an_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise ssl.SSLError("bad record mac")

        client = _client(handler, max_retries=0)

        with pytest.raises(ClassifierError, match="request failed"):
            await client.ask("a", NOUL)
        await client.close()

    @pytest.mark.asyncio
    async def test_no_answer_in_time_is_an_error(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(1)
            return _reply(_noul(0.5))

        client = _client(handler, timeout=0.05)

        with pytest.raises(ClassifierError, match="did not answer within 0.05s"):
            await client.ask("a", NOUL)
        await client.close()

    @pytest.mark.asyncio
    async def test_connect_looks_up_the_model(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((request.method, request.url.path))
            return _model(request)

        client = _client(handler, model="gemini-decision-next")
        await client.connect()

        assert seen == [("GET", "/v1beta/models/gemini-decision-next")]
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
    async def test_a_refused_connect_carries_googles_reason(self):
        client = _client(lambda request: _failure(404, "models/gemini-decision-1 is not found"))

        with pytest.raises(
            ClassifierError, match="could not connect.*HTTP 404: models/gemini-decision-1"
        ):
            await client.connect()
        await client.close()

    @pytest.mark.asyncio
    async def test_a_broken_tls_connection_at_connect_is_an_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise ssl.SSLError("bad record mac")

        client = _client(handler, max_retries=0)

        with pytest.raises(ClassifierError, match="could not connect"):
            await client.connect()
        await client.close()

    @pytest.mark.asyncio
    async def test_keeps_idle_connections_open_between_questions(self):
        client = GoogleDecisionsClient(api_key="key")
        assert client._http._transport._pool._keepalive_expiry == 240.0
        await client.close()

    @pytest.mark.asyncio
    async def test_close_closes_the_connection_pool(self):
        client = GoogleDecisionsClient(api_key="key")
        await client.close()
        assert client._http.is_closed

    def test_needs_an_api_key(self):
        with pytest.raises(ValueError):
            GoogleDecisionsClient(api_key="")


class TestGoogleDecisionsClassifier:
    @pytest.mark.asyncio
    async def test_yes_no(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["question"] = _question(request)
            return _reply(_noul(0.95))

        classifier = GoogleDecisionsClassifier(client=_client(handler))
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
    async def test_yes_no_says_what_counts_as_each_answer_in_its_criteria(self):
        questions = []

        def handler(request: httpx.Request) -> httpx.Response:
            questions.append(_question(request))
            return _reply(_noul(0.2))

        classifier = GoogleDecisionsClassifier(client=_client(handler))
        both = YesNoQuestion(instructions="voicemail?", yes="a recording", no="a person")
        await classifier.yes_no("Hello?", {"answer": both})
        await classifier.yes_no(
            "Hello?", {"answer": YesNoQuestion(instructions="voicemail?", yes="a recording")}
        )

        assert questions[0]["criteria"] == {
            "true_description": "a recording",
            "false_description": "a person",
        }
        assert questions[1]["criteria"] == {"true_description": "a recording"}
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_structured_state_and_instructions_are_sent_as_json(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen["input"] = body["input"]
            seen["instructions"] = body["questions"]["answer"]["instructions"]
            return _reply(_noul(0.5))

        classifier = GoogleDecisionsClassifier(client=_client(handler))
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
                    "choice": "technical",
                    "probabilities": {"billing": 0.16, "technical": 0.83},
                    "confidence": 0.83,
                }
            )

        classifier = GoogleDecisionsClassifier(client=_client(handler))
        options = {"billing": "payments", "technical": {"covers": ["outages"]}, "sales": None}
        result = (
            await classifier.choice(
                "Checkout is down",
                {"answer": ChoiceQuestion(instructions="which team?", options=options)},
            )
        )["answer"]

        assert result.choice == "technical"
        assert result.probabilities == {"billing": 0.16, "technical": 0.83, "sales": 0.0}
        assert result.confidence == 0.83
        assert seen["question"] == {
            "type": "choice",
            "instructions": "which team?",
            "criteria": {
                "billing": "payments",
                "technical": '{"covers": ["outages"]}',
                "sales": "",
            },
        }
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_too_many_options_is_an_error_before_any_request(self):
        sent = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            return _reply(_noul(1.0))

        classifier = GoogleDecisionsClassifier(client=_client(handler))
        options = {f"o{i}": None for i in range(GOOGLE_DECISIONS_MAX_CHOICE_OPTIONS + 1)}
        with pytest.raises(ClassifierError, match="at most 255 options"):
            await classifier.choice(
                "hmm", {"answer": ChoiceQuestion(instructions="?", options=options)}
            )
        assert sent == []
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_images_go_after_the_state(self):
        bodies = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return _reply(_noul(0.99))

        classifier = GoogleDecisionsClassifier(client=_client(handler))
        png = ClassifierImage(data=b"png bytes", content_type="image/png")
        question = {"answer": YesNoQuestion(instructions="is there a red circle?")}
        assert classifier.supports_images

        await classifier.yes_no("a camera frame", question, inputs=[png])
        await classifier.yes_no("no frame", question)

        assert bodies[0]["input"] == [
            {"type": "text", "text": "a camera frame"},
            {
                "type": "image",
                "data": base64.b64encode(b"png bytes").decode(),
                "mime_type": "image/png",
            },
        ]
        assert bodies[1]["input"] == "no frame"
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_a_clip_goes_where_it_is_given_as_16_khz_wav(self):
        bodies = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return _reply(_noul(0.1))

        classifier = GoogleDecisionsClassifier(client=_client(handler))
        png = ClassifierImage(data=b"png bytes", content_type="image/png")
        question = {"answer": YesNoQuestion(instructions="does the caller sound upset?")}
        assert classifier.supports_audio

        await classifier.yes_no("a caller", question, inputs=[_tone(1, sample_rate=8000), png])

        text, audio, image = bodies[0]["input"]
        assert text == {"type": "text", "text": "a caller"}
        assert image["type"] == "image"
        assert (audio["type"], audio["mime_type"]) == ("audio", "audio/wav")
        with wave.open(io.BytesIO(base64.b64decode(audio["data"]))) as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
            assert wav.getnframes() == pytest.approx(16000, abs=800)
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_each_file_goes_in_the_block_for_its_type(self):
        bodies = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return _reply(_noul(0.5))

        classifier = GoogleDecisionsClassifier(client=_client(handler))
        files = [
            ClassifierFile(content_type="application/pdf", data=b"%PDF-1.4"),
            ClassifierFile(content_type="video/mp4", url="https://example.com/clip.mp4"),
            ClassifierFile(content_type="audio/mp3", data=b"ID3"),
            ClassifierFile(content_type="image/gif", data=b"GIF89a"),
        ]

        await classifier.yes_no(
            "a ticket", {"answer": YesNoQuestion(instructions="?")}, inputs=files
        )

        assert bodies[0]["input"][1:] == [
            {"type": "document", "mime_type": "application/pdf", "data": _b64(b"%PDF-1.4")},
            {"type": "video", "mime_type": "video/mp4", "uri": "https://example.com/clip.mp4"},
            {"type": "audio", "mime_type": "audio/mp3", "data": _b64(b"ID3")},
            {"type": "image", "mime_type": "image/gif", "data": _b64(b"GIF89a")},
        ]
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_the_file_types_are_the_ones_google_reads(self):
        classifier = GoogleDecisionsClassifier(client=_client(lambda request: _reply(_noul(0.5))))

        assert {"application/pdf", "video/mp4", "audio/mp3", "image/gif"} <= classifier.file_types
        assert classifier.file_types == frozenset(GOOGLE_DECISIONS_FILE_TYPES)
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_a_file_of_another_type_is_refused_before_any_request(self):
        sent = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            return _reply(_noul(0.5))

        classifier = GoogleDecisionsClassifier(client=_client(handler))
        notes = ClassifierFile(content_type="text/plain", data=b"notes")

        with pytest.raises(ClassifierError, match="doesn't take text/plain files"):
            await classifier.yes_no(
                "notes", {"answer": YesNoQuestion(instructions="?")}, inputs=[notes]
            )
        assert sent == []
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_a_choice_outside_the_options_is_an_error(self):
        classifier = GoogleDecisionsClassifier(
            client=_client(
                lambda request: _reply({"type": "choice", "choice": "maybe", "confidence": 0.8})
            )
        )
        question = ChoiceQuestion(instructions="?", options={"yes": None, "no": None})
        with pytest.raises(ClassifierError, match="Google Decisions chose 'maybe'"):
            await classifier.choice("hmm", {"answer": question})
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_an_unusable_probability_is_an_error(self):
        classifier = GoogleDecisionsClassifier(
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
            seen["question"] = _question(request)
            return _reply(
                {
                    "type": "score",
                    "score": 1.97,
                    "legend": {"0": "calm", "1": "impatient", "2": '{"level": "angry"}'},
                    "probabilities": {"0": 0.01, "1": 0.01, "2": 0.98},
                    "confidence": 0.98,
                }
            )

        classifier = GoogleDecisionsClassifier(client=_client(handler))
        levels = ["calm", "impatient", {"level": "angry"}]
        result = (
            await classifier.score(
                "This is the third time!",
                {"answer": ScoreQuestion(instructions="how upset?", levels=levels)},
            )
        )["answer"]

        assert result.score == 1.97
        assert [(level.level, level.probability) for level in result.levels] == [
            ("calm", 0.01),
            ("impatient", 0.01),
            ({"level": "angry"}, 0.98),
        ]
        assert result.confidence == 0.98
        assert seen["question"] == {
            "type": "score",
            "instructions": "how upset?",
            "criteria": ["calm", "impatient", '{"level": "angry"}'],
        }
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_a_missing_answer_field_is_an_error(self):
        classifier = GoogleDecisionsClassifier(
            client=_client(lambda request: _reply({"type": "noul"}))
        )

        with pytest.raises(ClassifierError, match="noul"):
            await classifier.yes_no("a", {"answer": YesNoQuestion(instructions="?")})
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_an_answer_of_another_kind_is_an_error(self):
        classifier = GoogleDecisionsClassifier(
            client=_client(
                lambda request: _reply({"type": "choice", "choice": "yes", "confidence": 0.9})
            )
        )

        with pytest.raises(ClassifierError, match="noul"):
            await classifier.yes_no("a", {"answer": YesNoQuestion(instructions="?")})
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_ask_sends_every_question_in_one_request(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return _replies(
                {
                    "greeting": _noul(0.9),
                    "mood": {
                        "type": "score",
                        "score": 0.2,
                        "probabilities": {"0": 0.8, "1": 0.2},
                        "confidence": 0.8,
                    },
                }
            )

        classifier = GoogleDecisionsClassifier(client=_client(handler))
        results = await classifier.ask(
            "Hello there!",
            {
                "greeting": YesNoQuestion(instructions="is this a greeting?"),
                "mood": ScoreQuestion(instructions="how upset?", levels=["calm", "upset"]),
            },
        )

        assert len(requests) == 1
        assert {name: q["type"] for name, q in requests[0]["questions"].items()} == {
            "greeting": "noul",
            "mood": "score",
        }
        assert results["greeting"].probability == 0.9
        assert results["mood"].probability("calm") == 0.8
        await classifier.client.close()

    def test_needs_an_api_key_or_a_client(self):
        with pytest.raises(ValueError):
            GoogleDecisionsClassifier()

    @pytest.mark.asyncio
    async def test_a_client_of_its_own_takes_the_model_and_timeout(self):
        classifier = GoogleDecisionsClassifier(
            api_key="key", model="gemini-decision-next", timeout=1.5
        )
        assert classifier._owns_client
        assert classifier.model == "gemini-decision-next"
        assert classifier.client._http.timeout.read == 1.5
        await classifier.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup_closes_only_an_owned_client(self):
        owned = GoogleDecisionsClassifier(api_key="key")
        await owned.cleanup()
        assert owned.client._http.is_closed

        shared = GoogleDecisionsClient(api_key="key")
        classifier = GoogleDecisionsClassifier(client=shared)
        await classifier.cleanup()
        assert not shared._http.is_closed
        await shared.close()

    @pytest.mark.asyncio
    async def test_setup_opens_the_connection(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.path)
            return _model(request)

        classifier = GoogleDecisionsClassifier(client=_client(handler))
        await classifier.setup(TaskManager())

        assert seen == ["/v1beta/models/gemini-decision-1"]
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_a_failed_connect_at_setup_is_only_a_warning(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        classifier = GoogleDecisionsClassifier(client=_client(handler, max_retries=0))
        await classifier.setup(TaskManager())
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_a_shared_client_connects_once(self):
        connects = []

        def handler(request: httpx.Request) -> httpx.Response:
            connects.append(request.url.path)
            return _model(request)

        client = _client(handler)
        first = GoogleDecisionsClassifier(client=client)
        second = GoogleDecisionsClassifier(client=client)
        task_manager = TaskManager()
        await asyncio.gather(first.setup(task_manager), second.setup(task_manager))
        await first.setup(task_manager)

        assert len(connects) == 1
        await client.close()

    @pytest.mark.asyncio
    async def test_every_call_reports_its_time_and_tokens(self):
        client = _client(lambda request: _reply(_noul(0.9)))
        classifier = GoogleDecisionsClassifier(client=client, name="voicemail")
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
        assert (processing.processor, processing.model) == ("voicemail", "gemini-decision-1")
        assert isinstance(usage, LLMUsageMetricsData)
        assert (usage.value.prompt_tokens, usage.value.total_tokens) == (12, 12)
        await client.close()


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
class TestGoogleDecisionsLive:
    @pytest.mark.asyncio
    async def test_three_questions(self):
        classifier = GoogleDecisionsClassifier(api_key=os.environ["GOOGLE_API_KEY"])
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
        classifier = GoogleDecisionsClassifier(api_key=os.environ["GOOGLE_API_KEY"])
        question = {"red": YesNoQuestion(instructions="Is there a red circle in the image?")}
        try:
            red = await classifier.yes_no("A camera frame.", question, inputs=[_circle("red")])
            blue = await classifier.yes_no("A camera frame.", question, inputs=[_circle("blue")])
            assert red["red"].is_yes
            assert not blue["red"].is_yes
        finally:
            await classifier.cleanup()

    @pytest.mark.asyncio
    async def test_a_question_about_a_clip(self):
        classifier = GoogleDecisionsClassifier(api_key=os.environ["GOOGLE_API_KEY"])
        question = {"speech": YesNoQuestion(instructions="Is a person speaking in this clip?")}
        try:
            results = await classifier.yes_no("A clip of audio.", question, inputs=[_tone(2)])
            assert not results["speech"].is_yes
        finally:
            await classifier.cleanup()

    @pytest.mark.asyncio
    async def test_questions_about_a_pdf_and_a_video(self):
        classifier = GoogleDecisionsClassifier(api_key=os.environ["GOOGLE_API_KEY"])
        video = ClassifierFile(content_type="video/mp4", url=FIRE_VIDEO)
        questions = {
            "total": YesNoQuestion(instructions="Does the invoice say the total is 240 USD?"),
            "fire": YesNoQuestion(instructions="Does the video show fire or flames?"),
        }
        try:
            results = await classifier.yes_no(
                "An invoice and a video.", questions, inputs=[_invoice(), video]
            )
            assert results["total"].is_yes
            assert results["fire"].is_yes
        finally:
            await classifier.cleanup()
