#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for PlivoTTSService and PlivoHttpTTSService."""

import asyncio
import itertools
import json
from base64 import b64encode
from collections.abc import Iterator
from dataclasses import dataclass, field
from unittest.mock import AsyncMock

import aiohttp
import pytest
from loguru import logger
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedError, InvalidStatus
from websockets.http11 import Response
from websockets.protocol import State

from pipecat.frames.frames import (
    DataFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.plivo.tts import (
    PlivoHttpTTSService,
    PlivoTTSService,
    language_to_plivo_language,
)
from pipecat.services.tts_service import TextAggregationMode
from pipecat.tests.utils import run_test
from pipecat.transcriptions.language import Language

AUDIO = b"\x01\x00" * 480
MAX_TEXT = 20000  # the API's limit for the text of one context
UNKNOWN_VOICE = "unknown-voice"
PLIVO_LANGUAGES = set("bn en-AU en-GB en-IN en-US gu hi kn mr ta te".split())
# The fields that the API accepts in the text that opens a context.
PLIVO_FIELDS = set(
    "type context_id text finish timestamps model_id voice_id language pace volume "
    "pronunciation_dictionary_id seed continuation text_normalization "
    "use_normalized_timestamps output_format".split()
)
_TTS_FRAMES = (
    TTSStartedFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
    TTSStoppedFrame,
    LLMFullResponseEndFrame,
)


class _FakePlivo:
    """A fake Realtime API: audio and done for a finish or a cancel, an error for a refusal.

    As the API, it drops with a warning the text that takes a context past 20,000 characters.

    Attributes:
        spoken: The text of each context that got its finish, without the text dropped.
        before_done: Events sent between a context's audio and its done.
        answer_after: For a context's whole text, an event that its answer waits for.
        before_send: Awaited with each message, before the fake takes it.
        after_send: Awaited with each message, once the service has read its answers.
        drop_at: Drops the connection at the first message it returns True for.
        refusals: Exceptions, each raised by one connect.
        handshake: Each connect waits until it is set.
        connecting: Set when a connect starts to wait for ``handshake``.
    """

    def __init__(self):
        self.close = AsyncMock()
        self.ping = AsyncMock()
        self.state = State.OPEN
        self.sent: list[dict] = []
        self.spoken: list[str] = []
        self.connects: list[tuple] = []
        self.events: asyncio.Queue = asyncio.Queue()
        self.before_done: list[dict] = []
        self.answer_after: dict[str, asyncio.Event] = {}
        self.before_send = None
        self.after_send = None
        self.drop_at = None
        self.refusals: Iterator[Exception] = iter(())
        self.handshake = asyncio.Event()
        self.handshake.set()
        self.connecting = asyncio.Event()
        self._open_texts: dict[str, str] = {}
        self._text_chars: dict[str, int] = {}  # each context's text so far, by the API's count
        self._held: dict[str, asyncio.Task] = {}
        self._cancelled: list[str] = []
        self._ended: set[str] = set()

    async def connect(self, uri, **kwargs):
        """Stand in for ``websocket_connect``."""
        headers = kwargs["additional_headers"]
        self.connects.append((uri, headers))
        if refusal := next(self.refusals, None):
            raise refusal
        self.connecting.set()
        await asyncio.wait_for(self.handshake.wait(), 5)
        # As the API: every answer names its API ID, which starts with the client-ref.
        self.api_id = f"{headers.get('client-ref', 'plivo')}-api-1"
        self.response = Response(101, "", Headers({"api-id": self.api_id}))
        self.state = State.OPEN
        self._open_texts.clear()
        self._text_chars.clear()
        self._cancelled.clear()
        self._ended.clear()
        return self

    def drop(self):
        self.state = State.CLOSED
        self.events.put_nowait(None)

    async def send(self, data: str):
        message = json.loads(data)
        if self.before_send:
            await self.before_send(message)
        if self.drop_at and self.drop_at(message):
            self.drop_at = None
            self.drop()
        if self.state is State.CLOSED:
            raise ConnectionClosedError(None, None)
        self.sent.append(message)
        for cancelled in self._cancelled:
            self.end(cancelled, AUDIO, {"type": "done", "reason": "cancelled"})
        self._cancelled.clear()
        context_id = message["context_id"]

        if message["type"] == "cancel" and context_id in self._open_texts:
            if held := self._held.pop(context_id, None):
                held.cancel()
            self._cancelled.append(context_id)
        # An ended context's ID is used up on its connection: a message for it gets no answer.
        elif message["type"] == "text" and context_id not in self._ended:
            self._answer_text(context_id, message)
        await asyncio.wait_for(self.events.join(), 5)
        if self.after_send:
            await self.after_send(message)

    def _answer_text(self, context_id: str, message: dict):
        text = message.get("text", "")
        if context_id not in self._open_texts:
            error = {"type": "error", "code": "invalid_request", "status_code": 400}
            if unknown := sorted(message.keys() - PLIVO_FIELDS):
                return self.end(context_id, error | {"message": f"{unknown[0]}: unknown field"})
            if "language" in message and message["language"] not in PLIVO_LANGUAGES:
                return self.end(context_id, error | {"message": "language: not a Plivo code"})
            if message.get("voice_id") == UNKNOWN_VOICE:
                error = {"type": "error", "code": "voice_not_found", "status_code": 404}
                return self.end(context_id, error | {"message": "no voice has this ID"})
        taken = self._open_texts.get(context_id, "")
        # as the API: the text that takes a context past its limit is dropped with a warning,
        # and so is every text after that one
        self._text_chars[context_id] = self._text_chars.get(context_id, 0) + len(text)
        if text and self._text_chars[context_id] > MAX_TEXT:
            warning = {"type": "warning", "code": "text_too_long", "param": "text"}
            self.emit(context_id, warning | {"message": "text dropped"})
            text = ""
        text = taken + text
        self._open_texts[context_id] = text
        if message["finish"]:
            self.spoken.append(text)
            answer = (AUDIO, *self.before_done, {"type": "done", "reason": "complete"})
            if wait := self.answer_after.get(text):
                held = asyncio.create_task(self._end_after(wait, context_id, *answer))
                self._held[context_id] = held
            else:
                self.end(context_id, *answer)

    async def _end_after(self, wait: asyncio.Event, context_id: str, *events: bytes | dict):
        await wait.wait()
        del self._held[context_id]
        self.end(context_id, *events)

    def end(self, context_id: str, *events: bytes | dict):
        self._open_texts.pop(context_id, None)
        self._ended.add(context_id)
        self.emit(context_id, *events)

    def emit(self, context_id: str, *events: bytes | dict):
        header = len(context_id).to_bytes(2, "little") + context_id.encode()
        for event in events:
            if isinstance(event, bytes):
                self.events.put_nowait(header + event)
                continue
            if event["type"] != "timestamps":
                event = event | {"api_id": self.api_id}
            self.events.put_nowait(json.dumps(event | {"context_id": context_id}))

    async def __aiter__(self):
        """Give the service each message once it has handled the one before."""
        while (message := await self.events.get()) is not None:
            yield message
            self.events.task_done()
        self.state = State.CLOSED
        self.events.task_done()


class _FakeHttpSession:
    """A fake aiohttp session: ``AUDIO`` in two pieces that split a sample, or a refusal.

    Attributes:
        status: The status of each response.
        body: The body of a response whose status is not 200.
        body_error: Raised between the two pieces of audio.
    """

    def __init__(self):
        self.requests: list[tuple[str, dict, dict]] = []
        self.status = 200
        self.body = b""
        self.body_error: Exception | None = None

    def post(self, url, *, json, headers):
        self.requests.append((url, headers, json))
        api_id = f"{headers.get('client-ref', 'plivo')}-api-2"
        return _FakeHttpResponse(self.status, self.body, self.body_error, api_id)


class _FakeHttpResponse:
    def __init__(self, status: int, body: bytes, body_error: Exception | None, api_id: str):
        self.status = status
        self.content = self
        self.headers = {"api-id": api_id}
        self._body = body
        self._body_error = body_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *error):
        return False

    async def text(self, errors: str = "strict") -> str:
        return self._body.decode(errors=errors)

    async def iter_chunked(self, size: int):
        yield AUDIO[:481]
        if self._body_error:
            raise self._body_error
        yield AUDIO[481:]


@dataclass
class _WaitFrame(DataFrame):
    """Makes ``_Gate`` hold the frames after it until ``event`` is set, then interrupt if asked."""

    event: asyncio.Event = field(default_factory=asyncio.Event)
    interrupt: bool = False


class _Gate(FrameProcessor):
    """Passes each frame on to the service, and waits at a ``_WaitFrame`` for its event."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not isinstance(frame, _WaitFrame):
            await self.push_frame(frame, direction)
            return
        await asyncio.wait_for(frame.event.wait(), 5)
        if frame.interrupt:
            await self.push_frame(InterruptionFrame())


@pytest.fixture
def plivo(monkeypatch) -> _FakePlivo:
    plivo = _FakePlivo()
    monkeypatch.setattr("pipecat.services.websocket_service.websocket_connect", plivo.connect)
    return plivo


@pytest.fixture
def logged() -> Iterator[list[str]]:
    lines: list[str] = []
    sink = logger.add(lines.append, level="DEBUG")
    yield lines
    logger.remove(sink)


def _make_service(**kwargs) -> PlivoTTSService:
    settings = PlivoTTSService.Settings(voice="test-voice")
    return PlivoTTSService(auth_id="MA_TEST", auth_token="secret", settings=settings, **kwargs)


def _make_http_service(session, **kwargs) -> PlivoHttpTTSService:
    return PlivoHttpTTSService(
        auth_id="MA_TEST",
        auth_token="secret",
        aiohttp_session=session,
        settings=PlivoHttpTTSService.Settings(voice="test-voice"),
        **kwargs,
    )


def _after(service: PlivoTTSService, frame_type: type, pushed: bool = False) -> asyncio.Event:
    """An event that is set when the service has processed, or pushed, a frame of a type."""
    event = asyncio.Event()

    @service.event_handler("on_after_push_frame" if pushed else "on_after_process_frame")
    async def set_event(service, frame):
        if isinstance(frame, frame_type):
            event.set()

    return event


def _reconnected(plivo: _FakePlivo, service: PlivoTTSService) -> asyncio.Event:
    reconnected = asyncio.Event()

    @service.event_handler("on_connected")
    async def on_connected(service):
        if len(plivo.connects) > 1:
            reconnected.set()

    return reconnected


def _interruption(service: PlivoTTSService, after: asyncio.Event) -> list[Frame]:
    """Frames that interrupt the service once ``after`` is set, and wait until it has handled it."""
    return [_WaitFrame(after, interrupt=True), _WaitFrame(_after(service, InterruptionFrame))]


def _voice_update() -> TTSUpdateSettingsFrame:
    return TTSUpdateSettingsFrame(delta=PlivoTTSService.Settings(voice="other-voice"))


def _llm_response(*parts: str | Frame, ended: bool = True) -> list[Frame]:
    """The frames of an LLM response: each string is a token, in order with the frames given."""
    frames = [LLMTextFrame(part) if isinstance(part, str) else part for part in parts]
    end = [LLMFullResponseEndFrame()] if ended else []
    return [LLMFullResponseStartFrame(), *frames, *end]


def _timestamps(spoken: str, start_ms: int) -> dict:
    """A ``timestamps`` event for text spoken from ``start_ms``, at 10 ms a character."""
    starts = [start_ms + 10 * at for at in range(len(spoken))]
    times = {"start_ms": starts, "duration_ms": [10] * len(spoken)}
    return {"type": "timestamps", "chars": list(spoken)} | times


def _finish_message(first: dict) -> dict:
    return {"type": "text", "context_id": first["context_id"], "finish": True}


def _types(frames, *types: type) -> list[type]:
    return [type(frame) for frame in frames if isinstance(frame, types)]


def _texts(frames) -> list[str]:
    return [frame.text for frame in frames if isinstance(frame, TTSTextFrame)]


def _audio(frames) -> bytes:
    return b"".join(frame.audio for frame in frames if isinstance(frame, TTSAudioRawFrame))


def _played_contexts(frames) -> set[str]:
    return {frame.context_id for frame in frames if isinstance(frame, TTSAudioRawFrame)}


def _errors(frames, about: str = "") -> list[str]:
    errors = [frame.error for frame in frames if isinstance(frame, ErrorFrame)]
    return [error for error in errors if about in error]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_ref, headers",
    [(None, {}), ("call-42", {"client-ref": "call-42"})],
    ids=["no client_ref", "client_ref"],
)
async def test_protocol_roundtrip(plivo, logged, client_ref, headers):
    down, up = await run_test(
        _make_service(client_ref=client_ref),
        frames_to_send=[TTSSpeakFrame("Hello.")],
        pipeline_params=PipelineParams(audio_out_sample_rate=16000),
    )

    url = "wss://api.plivo.com/v1/Account/MA_TEST/TTS/Realtime"
    authorization = "Basic " + b64encode(b"MA_TEST:secret").decode()
    assert plivo.connects == [(url, {"Authorization": authorization} | headers)]
    assert any(f"Plivo API ID {plivo.api_id}" in line for line in logged)
    played = (TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame)
    started, audio, stopped = [frame for frame in down if isinstance(frame, played)]
    assert (audio.audio, audio.sample_rate, audio.num_channels) == (AUDIO, 16000, 1)
    assert started.context_id == audio.context_id == stopped.context_id
    assert _errors(down + up) == []
    plivo.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_token_streaming_sends_tokens_into_one_context(plivo):
    """Only the first message carries the settings, and the turn's end sends the finish."""
    _, up = await run_test(
        _make_service(),
        frames_to_send=_llm_response("Hello", " there."),
        pipeline_params=PipelineParams(audio_out_sample_rate=16000),
    )

    context_id = plivo.sent[0]["context_id"]
    text = {"type": "text", "context_id": context_id, "text": "Hello", "finish": False}
    output_format = {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000}
    settings = {
        "model_id": "koyal-v1",
        "voice_id": "test-voice",
        "output_format": output_format,
    }
    assert plivo.sent == [
        text | settings | {"timestamps": True},
        text | {"text": " there."},
        _finish_message(text),
    ]
    assert _errors(up) == []


@pytest.mark.asyncio
async def test_sentence_mode_flushes_each_sentence(plivo):
    """A flush has a sentence spoken without a wait for more text."""
    service = _make_service(text_aggregation_mode=TextAggregationMode.SENTENCE)

    await run_test(service, frames_to_send=_llm_response("One. ", "Two. ", "Three."))

    assert [message["type"] for message in plivo.sent] == ["text", "flush"] * 3 + ["text"]
    assert [message["text"] for message in plivo.sent[:-1:2]] == ["One. ", "Two. ", "Three. "]
    assert plivo.sent[-1] == _finish_message(plivo.sent[0])
    assert len({message["context_id"] for message in plivo.sent}) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, texts",
    [
        (TextAggregationMode.TOKEN, ["One. ", "Two. ", "Three."]),
        # Pipecat hands over the last sentence after the service has stopped.
        (TextAggregationMode.SENTENCE, ["One. ", "Two. ", "Three. "]),
    ],
)
async def test_text_frame_is_its_own_context(plivo, mode, texts):
    service = _make_service(text_aggregation_mode=mode)
    spoken = [TextFrame("One. "), TextFrame("Two. "), TextFrame("Three.")]

    _, up = await run_test(service, frames_to_send=[_voice_update(), *spoken])

    assert all(message["type"] == "text" and message["finish"] for message in plivo.sent)
    assert [message["text"] for message in plivo.sent] == texts
    assert len({message["context_id"] for message in plivo.sent}) == len(plivo.sent)
    assert plivo.sent[0]["voice_id"] == "other-voice"
    assert _errors(up) == []


@pytest.mark.asyncio
async def test_turn_without_text_sends_nothing(plivo):
    _, up = await run_test(_make_service(), frames_to_send=_llm_response())

    assert plivo.sent == []
    assert _errors(up) == []


@pytest.mark.asyncio
async def test_interruption_cancels_the_context(plivo):
    service = _make_service()
    frames = [
        *_llm_response("Hello", ended=False),
        *_interruption(service, after=_after(service, LLMTextFrame)),
        *_llm_response("Next."),
    ]

    down, _ = await run_test(Pipeline([_Gate(), service]), frames_to_send=frames)

    first, cancel, second, finish = plivo.sent
    assert cancel == {"type": "cancel", "context_id": first["context_id"]}
    assert second["context_id"] != first["context_id"]
    assert second["voice_id"] == "test-voice"
    assert finish["context_id"] == second["context_id"]
    assert len(plivo.connects) == 1
    # The audio that arrives for the cancelled context is not played.
    assert _played_contexts(down) == {second["context_id"]}


@pytest.mark.asyncio
async def test_interruption_of_two_contexts_drops_late_audio(plivo):
    """While Pipecat cancels two contexts in turn, audio that comes for the first is not played."""
    service = _make_service()
    # The speech has no answer yet when the interruption comes.
    plivo.answer_after["Please hold."] = asyncio.Event()
    frames = [
        *_llm_response("Hello there", TTSSpeakFrame("Please hold."), ended=False),
        *_interruption(service, after=_after(service, TTSSpeakFrame)),
    ]

    down, up = await run_test(Pipeline([_Gate(), service]), frames_to_send=frames)

    hello, speech, *_ = plivo.sent
    # The turn's context is cancelled first, and its audio comes with the second cancel.
    assert plivo.sent[-2:] == [
        {"type": "cancel", "context_id": hello["context_id"]},
        {"type": "cancel", "context_id": speech["context_id"]},
    ]
    assert _types(down, TTSAudioRawFrame) == []
    assert _errors(up) == []


@pytest.mark.asyncio
async def test_late_error_of_an_interrupted_context_is_skipped(plivo):
    """A cancelled context's late error is skipped: the uninterruptible speech plays in full."""
    service = _make_service()
    error = {"type": "error", "code": "backend_error", "status_code": 502}
    speech = TTSSpeakFrame("Important.")
    speech.interruptible = False
    plivo.before_done = [_timestamps("Important.", 0)]
    first_chunk_played = _after(service, TTSAudioRawFrame, pushed=True)

    async def hold_the_answers(message: dict):
        if message["type"] == "cancel":
            # The answer to the cancel comes later, with an error.
            plivo.end(message["context_id"])
        elif message["finish"]:
            # The rest of the speech comes once the next interruption is handled.
            plivo.answer_after["Important."] = _after(service, InterruptionFrame)

    async def play_a_chunk_then_end_hello(message: dict):
        if message.get("finish"):
            plivo.end(message["context_id"], AUDIO)
            await asyncio.wait_for(first_chunk_played.wait(), 5)
            plivo.end(plivo.sent[0]["context_id"], error | {"message": "the engine failed"})
            await asyncio.wait_for(plivo.events.join(), 5)

    plivo.before_send = hold_the_answers
    plivo.after_send = play_a_chunk_then_end_hello
    frames = [
        *_llm_response("Hello", ended=False),
        *_interruption(service, after=_after(service, LLMTextFrame)),
        speech,
        _WaitFrame(_after(service, TTSSpeakFrame), interrupt=True),
    ]

    down, up = await run_test(Pipeline([_Gate(), service]), frames_to_send=frames)

    assert _audio(down) == AUDIO * 2
    assert _texts(down) == ["Important."]
    assert service.get_audio_contexts() == []
    assert _errors(up) == []


@pytest.mark.asyncio
async def test_error_of_a_waiting_context_keeps_the_speech_playing(plivo):
    """An error for a context behind uninterruptible speech: the speech plays in full."""
    service = _make_service()
    speech = TTSSpeakFrame("Important.")
    speech.interruptible = False
    plivo.before_done = [_timestamps("Important.", 0)]
    # The rest of the speech comes once the interruption is handled.
    plivo.answer_after["Important."] = _after(service, InterruptionFrame)
    first_chunk_played = _after(service, TTSAudioRawFrame, pushed=True)

    async def play_a_chunk(message: dict):
        if message.get("finish") and message["context_id"] == plivo.sent[0]["context_id"]:
            plivo.end(message["context_id"], AUDIO)

    plivo.after_send = play_a_chunk
    refused = TTSUpdateSettingsFrame(delta=PlivoTTSService.Settings(voice=UNKNOWN_VOICE))
    frames = [
        speech,
        _WaitFrame(first_chunk_played),
        refused,
        TTSSpeakFrame("Next."),
        *_interruption(service, after=_after(service, ErrorFrame, pushed=True)),
    ]

    down, up = await run_test(Pipeline([_Gate(), service]), frames_to_send=frames)

    assert _errors(up) == ["Plivo error voice_not_found: no voice has this ID (API ID plivo-api-1)"]
    assert _audio(down) == AUDIO * 2
    assert _types(down, TTSStartedFrame, TTSStoppedFrame) == [TTSStartedFrame, TTSStoppedFrame]
    assert _texts(down) == ["Important."]
    assert service.get_audio_contexts() == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "delta, expected",
    [
        ({"voice": "other-voice"}, {"voice_id": "other-voice"}),
        ({"model": "koyal-v2"}, {"model_id": "koyal-v2"}),
        ({"language": Language.HI_IN}, {"language": "hi"}),
    ],
    ids=["voice", "model", "language"],
)
async def test_settings_update_rotates_context(plivo, delta, expected):
    """A context takes its settings from the text that opens it."""
    update = TTSUpdateSettingsFrame(delta=PlivoTTSService.Settings(**delta))

    down, up = await run_test(
        _make_service(), frames_to_send=_llm_response("Hello", update, " there.")
    )

    hello, finished, there, _ = plivo.sent
    assert (hello["text"], hello["voice_id"]) == ("Hello", "test-voice")
    assert finished == _finish_message(hello)
    assert there["context_id"] != hello["context_id"]
    assert there["text"] == " there." and expected.items() <= there.items()
    assert _texts(down) == ["Hello", "there."]
    assert len(_types(down, TTSAudioRawFrame)) == 2
    assert _errors(up) == []


@pytest.mark.asyncio
async def test_voice_change_keeps_word_order_and_times(plivo):
    """The old context plays first, with its words, even when the new one is answered first."""
    new_context_answered = plivo.answer_after["Hello"] = asyncio.Event()

    async def answer_new_context_first(message: dict):
        if message.get("text"):
            plivo.before_done = [_timestamps(message["text"], 0)]
        elif message["context_id"] != plivo.sent[0]["context_id"]:
            new_context_answered.set()

    plivo.before_send = answer_new_context_first

    down, up = await run_test(
        _make_service(), frames_to_send=_llm_response("Hello", _voice_update(), " there.")
    )

    old, new = plivo.sent[0]["context_id"], plivo.sent[2]["context_id"]
    spoken = [frame for frame in down if isinstance(frame, (TTSAudioRawFrame, TTSTextFrame))]
    assert [(type(frame), frame.context_id) for frame in spoken] == [
        (TTSAudioRawFrame, old),
        (TTSTextFrame, old),
        (TTSAudioRawFrame, new),
        (TTSTextFrame, new),
    ]
    hello, there = (frame for frame in spoken if isinstance(frame, TTSTextFrame))
    assert (hello.text, there.text) == ("Hello", "there.")
    # "Hello" ends at 50 ms, and "there." starts 10 ms into the new context, after its space.
    assert round((there.pts - hello.pts) / 1e6) == 60
    assert _errors(up) == []


@pytest.mark.asyncio
async def test_interruption_during_a_voice_change(plivo):
    service = _make_service()
    interrupted = _after(service, InterruptionFrame)

    async def interrupt_during_the_finish(message: dict):
        if message["type"] == "text" and "text" not in message:
            await service.queue_frame(InterruptionFrame())
            await asyncio.wait_for(interrupted.wait(), 5)

    plivo.before_send = interrupt_during_the_finish
    turn = _llm_response("Hello", _voice_update(), ended=False)
    frames = [*turn, _WaitFrame(interrupted), TextFrame("Outside.")]

    down, up = await run_test(Pipeline([_Gate(), service]), frames_to_send=frames)

    outside = plivo.sent[-1]
    assert (outside["text"], outside["finish"]) == ("Outside.", True)
    assert outside["voice_id"] == "other-voice"
    assert _played_contexts(down) == {outside["context_id"]}
    assert _errors(up) == []


@pytest.mark.asyncio
async def test_other_settings_apply_from_the_next_context(plivo):
    delta = PlivoTTSService.Settings(
        voice="test-voice",
        speed=1.2,
        volume=1.5,
        pronunciation_dictionary_id="support-terms",
        seed=0,
        continuation=False,
        text_normalization=False,
    )
    updated = {
        "voice_id": "test-voice",
        "pace": 1.2,
        "volume": 1.5,
        "pronunciation_dictionary_id": "support-terms",
        "seed": 0,
        "continuation": False,
        "text_normalization": False,
    }
    update = TTSUpdateSettingsFrame(delta=delta)

    frames = [*_llm_response("Hello", update, " there."), *_llm_response("Next.")]
    down, up = await run_test(_make_service(), frames_to_send=frames)

    hello, there, _, following, _ = plivo.sent
    assert there == {
        "type": "text",
        "context_id": hello["context_id"],
        "text": " there.",
        "finish": False,
    }
    assert {name: following[name] for name in updated} == updated
    assert _errors(up) == []


@pytest.mark.parametrize(
    "language, expected",
    [
        (Language.EN_AU, "en-AU"),
        (Language.EN_GB, "en-GB"),
        (Language.EN_IN, "en-IN"),
        (Language.EN_US, "en-US"),
        (Language.HI_IN, "hi"),
        # The API refuses ``en``.
        (Language.EN_NZ, "en"),
        (Language.EN, "en"),
    ],
)
def test_language_is_mapped_to_a_plivo_code(language, expected):
    assert language_to_plivo_language(language) == expected


@pytest.mark.asyncio
async def test_timestamps_become_timed_words(plivo):
    """A word starts at its first character's time, and a later sentence starts with a space."""
    plivo.before_done = [_timestamps("Hello there.", 0), _timestamps(" Bye now.", 500)]

    down, _ = await run_test(
        _make_service(), frames_to_send=_llm_response("Hello there.", " Bye now.")
    )

    words = [frame for frame in down if isinstance(frame, TTSTextFrame)]
    assert _texts(down) == ["Hello", "there.", "Bye", "now."]
    starts_ms = [round((word.pts - words[0].pts) / 1e6) for word in words]
    assert starts_ms == [0, 60, 510, 550]


@pytest.mark.asyncio
async def test_phoneme_span_is_one_timed_word(plivo):
    """A word that a pronunciation transform wrote as a ``{ }`` span is timed as that span.

    The context keeps the word as the LLM wrote it.
    """
    pronounce = PlivoTTSService.pronunciation_transform_ipa({"Plivo": "ˈpliːvoʊ"})
    plivo.before_done = [_timestamps("Welcome to {ˈpliːvoʊ}, friend.", 0)]

    down, _ = await run_test(
        _make_service(text_transforms=[("*", pronounce)]),
        frames_to_send=_llm_response("Welcome to Plivo, friend."),
    )

    words = [frame for frame in down if isinstance(frame, TTSTextFrame)]
    assert plivo.spoken == ["Welcome to {ˈpliːvoʊ}, friend."]
    assert [word.text for word in words] == ["Welcome", "to", "{ˈpliːvoʊ},", "friend."]
    assert " ".join(word.raw_text or word.text for word in words) == "Welcome to Plivo, friend."


@pytest.mark.asyncio
async def test_warning_is_logged_and_not_an_error(plivo, logged):
    """The text that takes a context past 20,000 characters is dropped, with a warning."""
    text = "word " * (MAX_TEXT // 5)

    down, up = await run_test(_make_service(), frames_to_send=_llm_response(text, "more "))

    assert plivo.spoken == [text]
    assert any("Plivo warning text_too_long: text dropped" in line for line in logged)
    assert _errors(up) == []
    assert _types(down, TTSStoppedFrame) == [TTSStoppedFrame]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "refused, frames, error",
    [
        ({"voice": UNKNOWN_VOICE}, [TextFrame("Hello")], "voice_not_found: no voice has this ID"),
        # The API refuses the text that opens the turn's context and gives the rest no answer. The
        # turn keeps its audio context until the idle timeout, so it is one context with no audio.
        (
            {"language": Language.EN},
            _llm_response("Hello", " there", " again."),
            "invalid_request: language: not a Plivo code",
        ),
    ],
    ids=["voice, a text outside a turn", "language, a turn"],
)
async def test_error_event_is_reported(plivo, refused, frames, error):
    bad = TTSUpdateSettingsFrame(delta=PlivoTTSService.Settings(**refused))
    good = TTSUpdateSettingsFrame(
        delta=PlivoTTSService.Settings(voice="test-voice", language=Language.EN_IN)
    )
    service = _make_service(stop_frame_timeout_s=0.5)

    down, up = await run_test(service, frames_to_send=[bad, *frames, good, *_llm_response("Next.")])

    assert _errors(up, "Plivo error") == [f"Plivo error {error} (API ID plivo-api-1)"]
    assert len(_errors(up, "completed with no audio")) == 1
    assert _audio(down) == AUDIO
    assert _types(down, TTSStartedFrame, TTSStoppedFrame) == [TTSStartedFrame, TTSStoppedFrame] * 2


@pytest.mark.asyncio
async def test_error_event_without_a_context_is_reported(plivo):
    """It is the connection's own error. The connection stays open unless the API closes it."""
    error = {"type": "error", "code": "internal_error", "status_code": 500, "api_id": "plivo-api-1"}
    plivo.events.put_nowait(json.dumps(error | {"message": "the server failed"}))

    down, up = await run_test(_make_service(), frames_to_send=[TextFrame("Hello there")])

    assert _errors(up, "Plivo error") == [
        "Plivo error internal_error: the server failed (API ID plivo-api-1)"
    ]
    assert _types(down, TTSAudioRawFrame) == [TTSAudioRawFrame]
    assert len(plivo.connects) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("lost_under", ["text", "flush"])
async def test_unsent_text_is_reported(plivo, lost_under):
    plivo.drop_at = lambda message: message["type"] == lost_under
    # In SENTENCE mode the sentence is sent while LLMFullResponseEndFrame is processed.
    service = _make_service(
        text_aggregation_mode=TextAggregationMode.SENTENCE, stop_frame_timeout_s=60
    )

    # The turn does not wait for the idle timeout of its audio context.
    async with asyncio.timeout(30):
        down, up = await run_test(service, frames_to_send=_llm_response("Hello."))

    assert _errors(up, "Plivo") == ["Plivo connection unavailable: message not sent"]
    assert _types(down, *_TTS_FRAMES) == [
        TTSStartedFrame,
        TTSStoppedFrame,
        LLMFullResponseEndFrame,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lost, reported",
    [
        # The send of " there." fails after the connection is replaced: it is still reported.
        ("before_send", ["Plivo connection unavailable: message not sent"]),
        # Lost after " there." was sent: no message failed, so none is reported.
        ("after_send", []),
    ],
    ids=["under a text", "between texts"],
)
async def test_turn_continues_on_a_new_connection(plivo, lost, reported):
    service = _make_service()
    reconnected = _reconnected(plivo, service)

    async def lose_the_connection(message: dict):
        if message.get("text") == " there.":
            plivo.drop()
            if lost == "before_send":
                await asyncio.wait_for(reconnected.wait(), 5)
                raise ConnectionClosedError(None, None)

    setattr(plivo, lost, lose_the_connection)
    frames = _llm_response("Hello", " there.", _WaitFrame(reconnected), " More.")

    down, up = await run_test(
        Pipeline([_Gate(), service]),
        frames_to_send=frames,
    )

    hello, *_, more, finished = plivo.sent
    assert len(plivo.connects) == 2
    assert (more["text"], more["voice_id"]) == (" More.", "test-voice")
    assert more["context_id"] != hello["context_id"]
    assert finished == _finish_message(more)
    assert _played_contexts(down) == {more["context_id"]}
    assert _errors(up, "Plivo") == reported


@pytest.mark.asyncio
async def test_new_connection_opens_the_turn_context_again(plivo):
    service = _make_service()
    reconnected = _reconnected(plivo, service)

    # Pipecat keeps the turn's context ID across a TTSSpeakFrame inside the turn, also when
    # the connection is replaced meanwhile.
    async def drop_after_speech(message: dict):
        if message.get("text") == "Please hold.":
            plivo.drop()
            await asyncio.wait_for(reconnected.wait(), 5)

    plivo.after_send = drop_after_speech
    frames = _llm_response("Hello", TTSSpeakFrame("Please hold."), " there.")

    down, _ = await run_test(service, frames_to_send=frames)

    hello, _, there, finished = plivo.sent
    assert len(plivo.connects) == 2
    assert there["context_id"] == hello["context_id"]
    assert (there["text"], there["voice_id"]) == (" there.", "test-voice")
    assert finished == _finish_message(there)
    assert _audio(down) == AUDIO


@pytest.mark.asyncio
@pytest.mark.parametrize("refused", [False, True], ids=["open context", "refused context"])
async def test_text_connects_after_the_loop_ends(plivo, refused):
    """With reconnect_on_error=False, Pipecat ends the receive loop when the connection closes."""
    service = _make_service(reconnect_on_error=False)
    error = {"type": "error", "code": "backend_error", "status_code": 502}
    # On the new connection, the turn's context is answered after the turn has ended.
    plivo.answer_after[" there."] = _after(service, LLMFullResponseEndFrame)
    loop_ended = []

    async def drop_and_wait_for_loop_end(message: dict):
        if message.get("text") == "Hello":
            if refused:
                plivo.end(message["context_id"], error | {"message": "the engine failed"})
            receive_loop = service._receive_task
            plivo.drop()
            done, _ = await asyncio.wait([receive_loop], timeout=5)
            loop_ended.append(receive_loop in done)

    plivo.after_send = drop_and_wait_for_loop_end

    down, up = await run_test(service, frames_to_send=_llm_response("Hello", " there."))

    assert loop_ended == [True]
    hello, there, finished = plivo.sent
    assert len(plivo.connects) == 2
    assert there["context_id"] == hello["context_id"]
    assert (there["text"], there["voice_id"]) == (" there.", "test-voice")
    assert finished == _finish_message(there)
    assert _audio(down) == AUDIO
    assert len(_errors(up, "connection closed by server")) == 1
    assert len(_errors(up, "Plivo error backend_error")) == refused


@pytest.mark.asyncio
async def test_interruption_on_a_lost_connection(plivo, logged):
    plivo.drop_at = lambda message: message["type"] == "cancel"
    service = _make_service()
    frames = [
        *_llm_response("Hello", ended=False),
        *_interruption(service, after=_after(service, LLMTextFrame)),
        _WaitFrame(_reconnected(plivo, service)),
        *_llm_response("Next."),
    ]

    down, up = await run_test(Pipeline([_Gate(), service]), frames_to_send=frames)

    assert not any("on_audio_context_interrupted failed" in line for line in logged)
    assert _errors(up) == []
    assert _audio(down) == AUDIO
    assert len(plivo.connects) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "refusal, connects, reason",
    [
        pytest.param(
            InvalidStatus(Response(401, "", Headers(), b'{"code": "unauthorized"}')),
            1,
            '{"code": "unauthorized"}',
            id="401",
        ),
        pytest.param(
            InvalidStatus(
                Response(429, "", Headers({"Retry-After": "60"}), b'{"code": "concurrency_limit"}')
            ),
            1,
            '{"code": "concurrency_limit"}',
            id="429 with Retry-After",
        ),
        pytest.param(EOFError(), 4, "", id="EOFError"),
    ],
)
async def test_refused_handshake_reports_its_reason(plivo, refusal, connects, reason):
    """A text tries to connect again, but not after a 401 or before the Retry-After time."""
    plivo.refusals = itertools.repeat(refusal)
    # The turn gets no audio, and a short idle timeout ends its audio context.
    service = _make_service(stop_frame_timeout_s=0.05)
    events = []

    @service.event_handler("on_connection_error")
    async def on_connection_error(service, error):
        events.append(error)

    _, up = await run_test(service, frames_to_send=_llm_response(*"012"))

    assert len(plivo.connects) == connects
    refusals = _errors(up, "Error connecting to Plivo")
    assert len(refusals) == connects and all(reason in refusal for refusal in refusals)
    assert len(events) == connects and all(reason in event for event in events)
    assert _errors(up, "processing frame") == []


@pytest.mark.asyncio
async def test_batch_request_posts_the_text_and_settings(logged):
    session = _FakeHttpSession()
    delta = PlivoHttpTTSService.Settings(
        language=Language.HI_IN,
        seed=0,
        continuation=False,
    )

    down, up = await run_test(
        _make_http_service(session, client_ref="call-42"),
        frames_to_send=[TTSUpdateSettingsFrame(delta=delta), TTSSpeakFrame("Hello.")],
        pipeline_params=PipelineParams(audio_out_sample_rate=16000),
    )

    body = {
        "text": "Hello.",
        "model_id": "koyal-v1",
        "voice_id": "test-voice",
        "language": "hi",
        "seed": 0,
        "continuation": False,
        "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000},
    }
    authorization = "Basic " + b64encode(b"MA_TEST:secret").decode()
    headers = {"Authorization": authorization, "client-ref": "call-42"}
    assert session.requests == [("https://api.plivo.com/v1/Account/MA_TEST/TTS", headers, body)]
    assert _audio(down) == AUDIO
    request = [frame for frame in down if isinstance(frame, _TTS_FRAMES)]
    assert [type(frame) for frame in request] == [
        TTSStartedFrame,
        TTSAudioRawFrame,
        TTSAudioRawFrame,
        TTSTextFrame,
        TTSStoppedFrame,
    ]
    assert _texts(down) == ["Hello."]
    assert len({frame.context_id for frame in request}) == 1
    assert _errors(up) == []
    assert any("Plivo API ID call-42-api-2" in line for line in logged)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body_error",
    [aiohttp.ClientPayloadError("Response payload is not completed"), TimeoutError()],
    ids=["ends early", "times out"],
)
async def test_batch_stream_failure_is_reported(body_error):
    """A body that breaks off or times out plays the audio that came, then reports one error."""
    session = _FakeHttpSession()
    session.body_error = body_error

    down, up = await run_test(_make_http_service(session), frames_to_send=[TTSSpeakFrame("Hello.")])

    assert _audio(down) == AUDIO[:480]
    assert len(_errors(up, "Plivo request or audio stream failed")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, body, reported",
    [
        (404, b'{"code": "voice_not_found"}', '{"code": "voice_not_found"}'),
        (502, b"Passerelle incorrecte \xe9", "Passerelle incorrecte \ufffd"),
    ],
    ids=["the API", "a proxy's page, not UTF-8"],
)
async def test_batch_error_response(status, body, reported):
    session = _FakeHttpSession()
    session.status, session.body = status, body
    service = _make_http_service(session, stop_frame_timeout_s=0.05)

    down, up = await run_test(service, frames_to_send=[TTSSpeakFrame("Hello.")])

    assert _errors(up, "Plivo error") == [f"Plivo error (status {status}): {reported}"]
    assert _types(down, TTSAudioRawFrame) == []
