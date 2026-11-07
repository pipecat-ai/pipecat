#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for AWSPollyTTSService word timestamps and its Polly client (issue #1015).

All Polly calls are mocked, so no live AWS credentials are needed.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import aiobotocore.session
import pytest
from botocore.exceptions import ClientError

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.aws.tts import AWSPollyTTSService
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.test_aws_credentials import _renewing_credentials, no_configured_keys  # noqa: F401

CTX = "ctx-1"

# A live speech marks response (newline-delimited JSON) captured from Polly for
# "Hello world from Pipecat". `time` is the word start offset in milliseconds;
# `start`/`end` are byte offsets into the input text (unused).
MARKS_SAMPLE = (
    b'{"time":6,"type":"word","start":0,"end":5,"value":"Hello"}\n'
    b'{"time":273,"type":"word","start":6,"end":11,"value":"world"}\n'
    b'{"time":612,"type":"word","start":12,"end":16,"value":"from"}\n'
    b'{"time":794,"type":"word","start":17,"end":24,"value":"Pipecat"}\n'
)


# --------------------------------------------------------------------------- #
# Test doubles for the aiobotocore Polly client
# --------------------------------------------------------------------------- #


class _FakeStream:
    """Stands in for a Polly StreamingBody with an async read()."""

    def __init__(self, data: bytes):
        self._data = data

    async def read(self) -> bytes:
        return self._data


class _FakePolly:
    """Fake Polly client: returns marks for json output, audio otherwise."""

    def __init__(self, pcm: bytes = b"", marks: bytes = b""):
        self._pcm = pcm
        self._marks = marks
        self.calls: list[dict] = []

    async def synthesize_speech(self, **params):
        self.calls.append(params)
        if params.get("OutputFormat") == "json":
            return {"AudioStream": _FakeStream(self._marks)}
        return {"AudioStream": _FakeStream(self._pcm)}


class _FakeClientCtx:
    """Async context manager returned by session.create_client()."""

    def __init__(self, session: "_FakeSession"):
        self._session = session

    async def __aenter__(self):
        self._session.opened += 1
        return self._session.polly

    async def __aexit__(self, *exc):
        self._session.closed += 1
        return False


class _FakeSession:
    """Fake aiobotocore session that counts the fake Polly clients it opens and closes."""

    def __init__(self, polly):
        self.polly = polly
        self.opened = 0
        self.closed = 0

    def create_client(self, *args, **kwargs):
        return _FakeClientCtx(self)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_service(**kwargs) -> AWSPollyTTSService:
    # Explicit credentials short-circuit the botocore chain (no network).
    # Word timestamps are on unless a test says otherwise.
    kwargs.setdefault("word_timestamps", True)
    return AWSPollyTTSService(
        api_key="test-secret",
        aws_access_key_id="test-key",
        region="us-east-1",
        **kwargs,
    )


def _pcm(seconds: float, sample_rate: int = 16000) -> bytes:
    """Return `seconds` of 16-bit mono silence at the given sample rate."""
    return b"\x00" * int(seconds * sample_rate * 2)


def _ndjson(words: list[tuple[str, int]]) -> bytes:
    """Build newline-delimited Polly word marks from (value, time_ms) pairs."""
    lines = [json.dumps({"time": t, "type": "word", "value": w}).encode() for w, t in words]
    return b"\n".join(lines) + b"\n"


def _use_polly(service: AWSPollyTTSService, polly):
    """Make ``polly`` the service's client, now and when setup opens one."""
    service._aws_session = _FakeSession(polly)
    service._client = polly


def _stub_for_run_tts(service: AWSPollyTTSService):
    """Stub sample rate, resampler and metrics so run_tts can run offline."""
    service._sample_rate = 16000
    service._task_manager = TaskManager()

    async def _aresample(data, in_rate, out_rate):
        return data

    async def _anoop(*args, **kwargs):
        pass

    service._resampler.resample = _aresample
    service.start_tts_usage_metrics = _anoop
    service.stop_ttfb_metrics = _anoop


def _capture_word_timestamps(service: AWSPollyTTSService) -> list:
    """Replace add_word_timestamps with a capturing stub; return the capture."""
    captured: list = []

    async def _fake(word_times, context_id=None, **kwargs):
        captured.extend(word_times)

    service.add_word_timestamps = _fake
    return captured


async def _drain(service: AWSPollyTTSService, text: str):
    return [frame async for frame in service.run_tts(text, CTX)]


# --------------------------------------------------------------------------- #
# Constructor wiring
# --------------------------------------------------------------------------- #


def test_word_timestamps_disabled_by_default():
    """Word timestamps are off by default, so Polly gets one request per sentence."""
    service = AWSPollyTTSService(api_key="test-secret", aws_access_key_id="test-key")
    assert service._word_timestamps is False
    assert service._push_text_frames is True


def test_word_timestamps_enabled_uses_per_word_path():
    """Enabling word timestamps activates the per-word path."""
    service = _make_service(word_timestamps=True)
    assert service._word_timestamps is True
    # Word events produce the TTSTextFrames, so the base must not push whole text.
    assert service._push_text_frames is False
    assert service._cumulative_time == 0.0


# --------------------------------------------------------------------------- #
# Speech marks fetch + parse
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fetch_word_marks_parses_ndjson():
    """Newline-delimited Polly marks parse into (word, start_seconds) pairs."""
    service = _make_service()
    polly = _FakePolly(marks=MARKS_SAMPLE)

    result = await service._fetch_word_marks(
        polly, {"Text": "x", "VoiceId": "Joanna", "SampleRate": "16000"}
    )

    assert result == [
        ("Hello", pytest.approx(0.006)),
        ("world", pytest.approx(0.273)),
        ("from", pytest.approx(0.612)),
        ("Pipecat", pytest.approx(0.794)),
    ]
    # The marks call asks for json word marks and drops SampleRate.
    sent = polly.calls[0]
    assert sent["OutputFormat"] == "json"
    assert sent["SpeechMarkTypes"] == ["word"]
    assert "SampleRate" not in sent


@pytest.mark.asyncio
async def test_word_marks_unescape_ssml_characters():
    """Marks for escaped characters keep every word of the sentence timed."""
    service = _make_service()
    # As Polly's neural engine returns them for this sentence.
    marks = _ndjson(
        [
            ("Call", 0),
            ("AT", 300),
            ("&amp;T", 700),
            ("if", 1000),
            ("x", 1100),
            ("&lt;", 1400),
            ("5", 1800),
        ]
    )
    _use_polly(service, _FakePolly(pcm=_pcm(1.0), marks=marks))

    down, _ = await run_test(
        service,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("Call AT&T if x < 5. "),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=2.5),
        ],
    )

    words = [f.raw_text for f in down if isinstance(f, TTSTextFrame)]
    assert words == ["Call", "AT&", "T", "if", "x", "<", "5."]


@pytest.mark.parametrize(
    "language,text,marks,expected",
    [
        (
            Language.JA,
            "今日はいい天気ですね。",
            [("今日", 0), ("は", 300), ("いい", 400), ("天気", 600), ("です", 900), ("ね", 1100)],
            "今日はいい天気ですね。",
        ),
        (
            Language.ZH,
            "今天天气很好。",
            [("今天", 0), ("天气", 400), ("很", 800), ("好", 1000)],
            "今天天气很好。",
        ),
        (
            Language.EN,
            "Hello there.",
            [("Hello", 0), ("there", 300)],
            "Hello there.",
        ),
    ],
)
@pytest.mark.asyncio
async def test_word_marks_keep_the_language_spacing_in_context(language, text, marks, expected):
    """Words of a language written without spaces reach the context unspaced."""
    service = _make_service(settings=AWSPollyTTSService.Settings(language=language))
    _use_polly(service, _FakePolly(pcm=_pcm(1.5), marks=_ndjson(marks)))
    context = LLMContext()

    await run_test(
        Pipeline([service, LLMAssistantAggregator(context)]),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame(text),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=2.5),
        ],
    )

    assert [m["content"] for m in context.get_messages()] == [expected]


@pytest.mark.asyncio
async def test_fetch_word_marks_returns_empty_on_failure():
    """A Polly error or a missing AudioStream yields [] and never raises."""
    service = _make_service()

    class _RaisingPolly:
        async def synthesize_speech(self, **params):
            raise ClientError({"Error": {"Code": "Boom", "Message": "boom"}}, "SynthesizeSpeech")

    class _NoStreamPolly:
        async def synthesize_speech(self, **params):
            return {}

    assert await service._fetch_word_marks(_RaisingPolly(), {}) == []
    assert await service._fetch_word_marks(_NoStreamPolly(), {}) == []


# --------------------------------------------------------------------------- #
# run_tts behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cumulative_offset_across_sentences():
    """A later sentence in the same turn is offset by the prior audio duration."""
    service = _make_service()
    _stub_for_run_tts(service)
    captured = _capture_word_timestamps(service)

    # Sentence 1: 1.0s of audio (32000 bytes), word "Hello" at 0 ms.
    _use_polly(service, _FakePolly(pcm=_pcm(1.0), marks=_ndjson([("Hello", 0)])))
    await _drain(service, "Hello")

    # Sentence 2: word "World" at 100 ms -> 0.1 + 1.0 (prior duration) = 1.1s.
    _use_polly(service, _FakePolly(pcm=_pcm(0.5), marks=_ndjson([("World", 100)])))
    await _drain(service, "World")

    assert captured == [("Hello", pytest.approx(0.0)), ("World", pytest.approx(1.1))]


@pytest.mark.asyncio
async def test_cancelling_run_tts_cancels_pending_marks_call():
    """Cancelling run_tts (an interruption) cancels the marks call still in flight."""
    service = _make_service()
    _stub_for_run_tts(service)
    captured = _capture_word_timestamps(service)

    marks_started = asyncio.Event()
    marks_cancelled = asyncio.Event()

    class _SlowMarksPolly(_FakePolly):
        async def synthesize_speech(self, **params):
            if params.get("OutputFormat") == "json":
                marks_started.set()
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    marks_cancelled.set()
                    raise
            return await super().synthesize_speech(**params)

    _use_polly(service, _SlowMarksPolly(pcm=_pcm(1.0)))
    drain = asyncio.create_task(_drain(service, "Hello"))
    await marks_started.wait()
    drain.cancel()
    with pytest.raises(asyncio.CancelledError):
        await drain

    assert marks_cancelled.is_set()
    assert captured == []


@pytest.mark.asyncio
async def test_no_marks_pushes_sentence_once():
    """Without word marks the sentence still reaches the context, exactly once."""
    service = _make_service()
    _use_polly(service, _FakePolly(pcm=_pcm(0.5), marks=b""))

    down, _ = await run_test(
        service,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello world. "),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=2.5),
        ],
    )

    assert [f.text for f in down if isinstance(f, TTSTextFrame)] == ["Hello world."]


@pytest.mark.parametrize("engine", ["neural", "generative"])
@pytest.mark.asyncio
async def test_untimed_sentence_reaches_context_before_an_interruption(engine):
    """A sentence played without marks stays in the context when the next one is interrupted."""
    service = _make_service(settings=AWSPollyTTSService.Settings(engine=engine))
    _use_polly(service, _FakePolly(pcm=_pcm(0.3), marks=b""))
    context = LLMContext()

    await run_test(
        Pipeline([service, LLMAssistantAggregator(context)]),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("One two. "),
            SleepFrame(sleep=1.0),
            LLMTextFrame("Three four. "),
            SleepFrame(sleep=0.1),
            InterruptionFrame(),
            SleepFrame(sleep=2.5),
        ],
    )

    assert [m["content"] for m in context.get_messages()] == ["One two."]


@pytest.mark.asyncio
async def test_word_timestamps_disabled_skips_marks_call():
    """With word timestamps off, run_tts never makes the json marks call."""
    service = _make_service(word_timestamps=False)
    _stub_for_run_tts(service)
    captured = _capture_word_timestamps(service)
    polly = _FakePolly(pcm=_pcm(0.5), marks=MARKS_SAMPLE)
    _use_polly(service, polly)

    await _drain(service, "Hello world")

    assert captured == []
    assert all(call.get("OutputFormat") != "json" for call in polly.calls)


@pytest.mark.asyncio
async def test_generative_engine_times_the_sentence_as_a_whole():
    """The generative engine has no speech marks, so its sentence is timed as a whole."""
    service = _make_service(settings=AWSPollyTTSService.Settings(engine="generative"))
    _stub_for_run_tts(service)
    captured = _capture_word_timestamps(service)
    polly = _FakePolly(pcm=_pcm(0.5), marks=MARKS_SAMPLE)
    _use_polly(service, polly)

    await _drain(service, "Hello world")

    assert captured == [("Hello world", pytest.approx(0.0))]
    assert all(call.get("OutputFormat") != "json" for call in polly.calls)


@pytest.mark.asyncio
async def test_generative_engine_warns_once_that_words_are_untimed():
    """Asking for word timestamps on the generative engine logs one warning."""
    service = _make_service(settings=AWSPollyTTSService.Settings(engine="generative"))
    _stub_for_run_tts(service)
    _use_polly(service, _FakePolly(pcm=_pcm(0.5)))

    with patch("pipecat.services.aws.tts.logger") as logger:
        await _drain(service, "Hello")
        await _drain(service, "World")

    assert logger.warning.call_count == 1


# --------------------------------------------------------------------------- #
# Offset reset
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("frame", [TTSStoppedFrame(context_id=CTX), InterruptionFrame()])
@pytest.mark.asyncio
async def test_cumulative_time_resets_on_end_of_turn_or_interruption(frame):
    """A TTSStoppedFrame (end of turn) or an interruption clears the accumulated offset."""
    service = _make_service()

    async def _noop_super(self, frame, direction=FrameDirection.DOWNSTREAM):
        pass

    service._cumulative_time = 5.0
    # Stub the parent push so the override runs without a wired pipeline.
    with patch.object(TTSService, "push_frame", _noop_super):
        await service.push_frame(frame)

    assert service._cumulative_time == 0.0


# --------------------------------------------------------------------------- #
# Polly client lifecycle
# --------------------------------------------------------------------------- #


async def _set_up(service: AWSPollyTTSService):
    await service.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=TaskManager(),
            pipeline_worker=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
        )
    )


async def _make_set_up_service() -> tuple[AWSPollyTTSService, _FakeSession]:
    service = _make_service(word_timestamps=False)
    session = _FakeSession(_FakePolly(pcm=_pcm(0.1)))
    service._aws_session = session
    await _set_up(service)
    service._sample_rate = 16000
    return service, session


@pytest.mark.asyncio
async def test_sentences_share_one_polly_client():
    """Setup opens one Polly client and every sentence goes through it."""
    service, session = await _make_set_up_service()

    for text in ("One.", "Two.", "Three."):
        await _drain(service, text)

    assert session.opened == 1
    assert session.closed == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["stop", "cancel", "cleanup"])
async def test_stopping_closes_the_polly_client(how):
    """Stopping, cancelling or cleaning up closes the client, once."""
    service, session = await _make_set_up_service()
    await _drain(service, "One.")

    if how == "stop":
        await service.stop(EndFrame())
    elif how == "cancel":
        await service.cancel(CancelFrame())
    else:
        await service.cleanup()
    await service.cleanup()

    assert session.closed == 1


@pytest.mark.asyncio
async def test_shared_client_signs_with_renewed_credentials(no_configured_keys):  # noqa: F811
    """Without configured keys, the shared client still renews its credentials."""
    service = AWSPollyTTSService(region="us-east-1")
    service._aws_session = aiobotocore.session.get_session()
    service._aws_session._credentials = _renewing_credentials()
    await _set_up(service)

    client = service._client
    signing_keys = [
        (await client._request_signer._credentials.get_frozen_credentials()).access_key
        for _ in range(2)
    ]
    await service.cleanup()

    assert signing_keys[0] != signing_keys[1]
