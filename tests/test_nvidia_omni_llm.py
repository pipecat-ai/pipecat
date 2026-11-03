#
# Copyright (c) 2026, Daily
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the NVIDIA Nemotron Omni LLM service and adapter."""

import asyncio
import base64
import contextlib
import io
import wave
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import pytest_asyncio
from openai import BadRequestError

from pipecat.adapters.llm.base_llm_adapter import LLMContextConversionError
from pipecat.frames.frames import (
    CancelFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.omni.llm import (
    NvidiaOmniLLMService,
    _audio_part,
    _TranscriptResponseExtractor,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.http import TIMEOUT_EXCEPTIONS


def _service(*, audio_input: bool = True, **settings) -> NvidiaOmniLLMService:
    return NvidiaOmniLLMService(
        base_url="http://localhost:8000/v1",
        audio_input=audio_input,
        settings=NvidiaOmniLLMService.Settings(**settings) if settings else None,
    )


def _tool_messages(content="sunny", tool_call_id="call_1"):
    return [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": "w", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": tool_call_id, "content": content},
    ]


def _chunk(content=None, *, tool_calls=None, reasoning_content=None):
    delta = SimpleNamespace(
        content=content, tool_calls=tool_calls, reasoning_content=reasoning_content
    )
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None, model=None)


async def _stream(*chunks):
    for chunk in chunks:
        yield chunk


class _Turns:
    """Replaces the service's turn runner with turns the test releases by hand."""

    def __init__(self, service: NvidiaOmniLLMService):
        self.started: list[dict] = []
        self.cancelled: list[int] = []
        self.completed: list[int] = []
        self.on_start: Callable[[], Awaitable[None]] | None = None
        self._releases: list[asyncio.Event] = []

        async def run_turn(context, **kwargs):
            index = len(self.started)
            release = asyncio.Event()
            self.started.append({"context": context, **kwargs})
            self._releases.append(release)
            if self.on_start:
                await self.on_start()
            try:
                await release.wait()
                self.completed.append(index)
            except asyncio.CancelledError:
                self.cancelled.append(index)
                raise

        async def cancel_task(task, timeout=None):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        service._run_turn = run_turn
        service.create_task = lambda coro, name=None: asyncio.create_task(coro, name=name)
        service.cancel_task = cancel_task
        service.stop_all_metrics = AsyncMock()

    def release(self, index: int):
        self._releases[index].set()

    async def wait_for(self, count: int):
        for _ in range(200):
            if len(self.started) >= count:
                return
            await asyncio.sleep(0.005)
        raise AssertionError(f"expected {count} turns, got {len(self.started)}")


def _fill_audio(service: NvidiaOmniLLMService, seconds: float = 1.0):
    service._audio_buffer = [b"\x00" * int(service._bytes_per_second() * seconds)]


@pytest_asyncio.fixture
async def omni():
    service = _service()
    turns = _Turns(service)
    yield service, turns
    for index in range(len(turns.started)):
        turns.release(index)
    await service._cancel_turns()


@pytest.mark.asyncio
async def test_speech_between_user_turn_frames_starts_an_audio_turn(omni):
    service, turns = omni
    service.push_frame = AsyncMock()
    audio = b"\x00" * service._bytes_per_second()

    await service.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await service.process_frame(
        InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )
    await service.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    await turns.wait_for(1)
    assert turns.started[0]["expect_transcript"] is True
    assert turns.started[0]["turn_parts"][0]["type"] == "input_audio"


async def _assert_queued_behind_the_first_turn(turns: _Turns):
    await asyncio.sleep(0.01)
    assert len(turns.started) == 1
    assert turns.cancelled == []

    turns.release(0)
    await turns.wait_for(2)
    assert turns.completed == [0]


@pytest.mark.asyncio
async def test_audio_turn_waits_for_the_turn_in_progress(omni):
    service, turns = omni
    _fill_audio(service)
    await service._start_audio_turn()
    await turns.wait_for(1)

    _fill_audio(service)
    await service._start_audio_turn()

    await _assert_queued_behind_the_first_turn(turns)


@pytest.mark.asyncio
async def test_text_turn_waits_for_the_turn_in_progress(omni):
    service, turns = omni
    context = LLMContext([{"role": "user", "content": "hi"}])
    await service._start_text_turn(context, force=True)
    await turns.wait_for(1)

    await service._start_text_turn(context, force=True)

    await _assert_queued_behind_the_first_turn(turns)


@pytest.mark.asyncio
async def test_context_frame_during_an_audio_turn_is_answered_after_it(omni):
    service, turns = omni
    _fill_audio(service)
    await service._start_audio_turn()
    await turns.wait_for(1)

    await service._start_text_turn(
        LLMContext([{"role": "developer", "content": "Mention the opening hours."}])
    )

    await _assert_queued_behind_the_first_turn(turns)


@pytest.mark.asyncio
async def test_interruption_cancels_every_queued_turn(omni):
    service, turns = omni
    service.push_frame = AsyncMock()
    await service._start_text_turn(LLMContext([{"role": "user", "content": "hi"}]), force=True)
    await turns.wait_for(1)
    await service._start_text_turn(LLMContext(_tool_messages()))

    await service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.01)

    assert turns.cancelled == [0]
    assert len(turns.started) == 1


@pytest.mark.asyncio
async def test_failed_turn_does_not_block_the_queue(omni):
    service, turns = omni
    failing = asyncio.get_running_loop().create_future()
    failing.set_exception(RuntimeError("boom"))
    service._turn_tasks = [failing]

    await service._start_text_turn(LLMContext([{"role": "user", "content": "hi"}]), force=True)
    await turns.wait_for(1)


@pytest.mark.asyncio
async def test_short_utterance_is_ignored(omni):
    service, turns = omni
    _fill_audio(service)
    await service._start_audio_turn()
    await turns.wait_for(1)

    _fill_audio(service, seconds=0.05)
    await service._start_audio_turn()
    await asyncio.sleep(0.01)

    assert len(turns.started) == 1
    assert turns.cancelled == []


@pytest.mark.asyncio
async def test_utterance_before_any_context_is_answered_without_history(omni):
    service, turns = omni
    _fill_audio(service)
    await service._start_audio_turn()
    await turns.wait_for(1)

    assert turns.started[0]["context"].get_messages() == []


@pytest.mark.asyncio
async def test_typed_user_turn_is_completed_with_audio_input(omni):
    service, turns = omni
    await service._start_text_turn(LLMContext([{"role": "user", "content": "hi"}]))
    await turns.wait_for(1)


@pytest.mark.asyncio
async def test_tool_result_is_completed_once_across_context_frames(omni):
    service, turns = omni
    context = LLMContext(_tool_messages())
    await service._start_text_turn(context)
    await turns.wait_for(1)
    turns.release(0)

    await service._start_text_turn(context)
    await asyncio.sleep(0.01)

    assert len(turns.started) == 1


@pytest.mark.asyncio
async def test_follow_up_waits_for_a_running_tool(omni):
    service, turns = omni
    await service._start_text_turn(LLMContext(_tool_messages(content="IN_PROGRESS")))
    assert not service._turn_tasks

    await service._start_text_turn(LLMContext(_tool_messages()))
    await turns.wait_for(1)


@pytest.mark.asyncio
async def test_async_tool_result_replacing_its_started_message_is_completed(omni):
    service, turns = omni
    await service._start_text_turn(LLMContext(_tool_messages(content="started")))
    await turns.wait_for(1)
    turns.release(0)

    await service._start_text_turn(LLMContext(_tool_messages(content="sunny")))
    await turns.wait_for(2)


@pytest.mark.asyncio
async def test_developer_instruction_is_completed_with_audio_input(omni):
    service, turns = omni
    context = LLMContext([{"role": "developer", "content": "Introduce yourself."}])
    await service._start_text_turn(context)
    await turns.wait_for(1)


@pytest.mark.asyncio
async def test_answered_context_is_not_completed(omni):
    service, _ = omni
    context = LLMContext(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    )
    await service._start_text_turn(context)
    assert not service._turn_tasks


async def _report_transcript(service: NvidiaOmniLLMService, transcript: str):
    service.push_frame = AsyncMock()
    await service._emit_user_transcript(transcript)


@pytest.mark.asyncio
async def test_context_frame_recording_a_spoken_turn_is_not_answered_again(omni):
    service, _ = omni
    await _report_transcript(service, "where is the tower?")
    await service._start_text_turn(LLMContext([{"role": "user", "content": "where is the tower?"}]))
    assert not service._turn_tasks


@pytest.mark.asyncio
async def test_later_frame_repeating_the_spoken_turn_is_answered(omni):
    service, turns = omni
    await _report_transcript(service, "where is the tower?")
    context = LLMContext([{"role": "user", "content": "where is the tower?"}])
    await service._start_text_turn(context)

    await service._start_text_turn(context)
    await turns.wait_for(1)


@pytest.mark.asyncio
async def test_developer_message_after_a_spoken_turn_is_answered(omni):
    service, turns = omni
    await _report_transcript(service, "where is the tower?")
    context = LLMContext(
        [
            {"role": "user", "content": "where is the tower?"},
            {"role": "developer", "content": "Mention the opening hours."},
        ]
    )
    await service._start_text_turn(context)
    await turns.wait_for(1)


@pytest.mark.asyncio
async def test_late_spoken_turn_frame_and_tool_result_frame_complete_the_result_once(omni):
    service, turns = omni
    await _report_transcript(service, "weather?")
    context = LLMContext(_tool_messages())
    await service._start_text_turn(context)
    await turns.wait_for(1)
    turns.release(0)

    await service._start_text_turn(context)
    await asyncio.sleep(0.01)

    assert len(turns.started) == 1


@pytest.mark.asyncio
async def test_tool_follow_up_carries_a_spoken_turn_missing_from_the_context(omni):
    service, turns = omni
    service._last_transcript = "what is the weather?"
    await service._start_text_turn(LLMContext(_tool_messages()[1:]))
    await turns.wait_for(1)

    assert turns.started[0]["turn_parts"] == [{"type": "text", "text": "what is the weather?"}]


@pytest.mark.asyncio
async def test_tool_follow_up_waits_for_the_turn_that_requested_it(omni):
    service, turns = omni
    await service._start_text_turn(LLMContext([{"role": "user", "content": "hi"}]), force=True)
    await turns.wait_for(1)

    await service._start_text_turn(LLMContext(_tool_messages()))
    await asyncio.sleep(0.01)
    assert len(turns.started) == 1
    assert turns.cancelled == []

    turns.release(0)
    await turns.wait_for(2)
    assert turns.completed == [0]


@pytest.mark.asyncio
async def test_interruption_cancels_the_turn_and_drops_buffered_audio(omni):
    service, turns = omni
    service.push_frame = AsyncMock()
    _fill_audio(service)
    await service._start_audio_turn()
    await turns.wait_for(1)
    service._pre_speech_buffer = [b"\x00" * 320]

    await service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    assert turns.cancelled == [0]
    assert not service._turn_tasks
    assert service._pre_speech_buffer == []
    service.stop_all_metrics.assert_awaited()


async def _speak(service: NvidiaOmniLLMService, *, start_turn: bool = True, seconds: float = 1.0):
    frames = [VADUserStartedSpeakingFrame()]
    if start_turn:
        frames.append(UserStartedSpeakingFrame())
    audio = b"\x00" * int(service._bytes_per_second() * seconds)
    frames += [
        InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1),
        VADUserStoppedSpeakingFrame(),
    ]
    if start_turn:
        frames.append(UserStoppedSpeakingFrame())
    for frame in frames:
        await service.process_frame(frame, FrameDirection.DOWNSTREAM)


async def _speak_over_the_bot(service: NvidiaOmniLLMService, audio: bytes):
    for frame in [
        VADUserStartedSpeakingFrame(),
        UserStartedSpeakingFrame(),
        InterruptionFrame(),
        InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1),
        VADUserStoppedSpeakingFrame(),
        UserStoppedSpeakingFrame(),
    ]:
        await service.process_frame(frame, FrameDirection.DOWNSTREAM)


def _sent_audio(turn: dict) -> bytes:
    wav = base64.b64decode(turn["turn_parts"][0]["input_audio"]["data"])
    with wave.open(io.BytesIO(wav)) as reader:
        return reader.readframes(reader.getnframes())


@pytest.mark.asyncio
async def test_speech_interrupted_before_its_transcript_is_answered_with_the_next_utterance(omni):
    service, turns = omni
    service.push_frame = AsyncMock()
    first = b"\x01" * service._bytes_per_second()
    second = b"\x02" * service._bytes_per_second()
    service._audio_buffer = [first]
    await service._start_audio_turn()
    await turns.wait_for(1)

    await _speak_over_the_bot(service, second)
    await turns.wait_for(2)

    assert turns.cancelled == [0]
    assert _sent_audio(turns.started[1]) == first + second


@pytest.mark.asyncio
async def test_transcribed_speech_is_not_answered_again_after_an_interruption(omni):
    service, turns = omni
    turns.on_start = lambda: _report_transcript(service, "What's the weather?")
    second = b"\x02" * service._bytes_per_second()
    _fill_audio(service)
    await service._start_audio_turn()
    await turns.wait_for(1)

    await _speak_over_the_bot(service, second)
    await turns.wait_for(2)

    assert _sent_audio(turns.started[1]) == second


@pytest.mark.asyncio
async def test_speech_of_a_finished_turn_is_not_answered_again(omni):
    service, turns = omni
    service.push_frame = AsyncMock()
    second = b"\x02" * service._bytes_per_second()
    _fill_audio(service)
    await service._start_audio_turn()
    await turns.wait_for(1)
    turns.release(0)
    await asyncio.sleep(0.01)

    await _speak_over_the_bot(service, second)
    await turns.wait_for(2)

    assert _sent_audio(turns.started[1]) == second


@pytest.mark.asyncio
async def test_speech_without_an_interruption_lets_the_reply_finish(omni):
    service, turns = omni
    service.push_frame = AsyncMock()
    _fill_audio(service)
    await service._start_audio_turn()
    await turns.wait_for(1)

    await _speak(service)

    await _assert_queued_behind_the_first_turn(turns)


@pytest.mark.asyncio
async def test_speech_that_starts_no_user_turn_is_dropped(omni):
    service, turns = omni
    service.push_frame = AsyncMock()
    await _speak(service, start_turn=False, seconds=2.0)
    await service.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    assert not service._turn_tasks

    await _speak(service)
    await turns.wait_for(1)

    sent_audio = base64.b64decode(turns.started[0]["turn_parts"][0]["input_audio"]["data"])
    assert len(sent_audio) < 2 * service._bytes_per_second()


@pytest.mark.asyncio
async def test_pre_speech_buffer_keeps_only_the_most_recent_audio(omni):
    service, _ = omni
    chunk = b"\x00" * (service._bytes_per_second() // 10)
    for _ in range(10):
        service._handle_input_audio(
            InputAudioRawFrame(audio=chunk, sample_rate=16000, num_channels=1)
        )

    retained = sum(len(c) for c in service._pre_speech_buffer)
    assert retained <= service._bytes_per_second() * service._settings.pre_speech_buffer_secs
    assert retained > 0


@pytest.mark.asyncio
async def test_run_frame_completes_an_answered_context(omni):
    service, turns = omni
    service._context = LLMContext(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    )
    await service.process_frame(LLMRunFrame(), FrameDirection.DOWNSTREAM)
    await turns.wait_for(1)


@pytest.mark.asyncio
async def test_completion_before_any_context_is_ignored(omni):
    service, _ = omni
    await service._start_text_turn(None, force=True)
    assert not service._turn_tasks


@pytest.mark.asyncio
async def test_context_without_a_user_turn_is_not_completed(omni):
    service, _ = omni
    context = LLMContext([LLMSpecificMessage(llm="other", message={"role": "user"})])
    await service._start_text_turn(context)
    assert not service._turn_tasks


@pytest.mark.asyncio
async def test_cancel_frame_cancels_the_turn_in_progress(omni):
    service, turns = omni
    _fill_audio(service)
    await service._start_audio_turn()
    await turns.wait_for(1)

    await service.cancel(CancelFrame())

    assert turns.cancelled == [0]


@pytest.mark.asyncio
async def test_custom_audio_response_instruction_is_sent(omni):
    service, turns = omni
    service._settings.audio_response_instruction = "Answer in one word."
    _fill_audio(service)
    await service._start_audio_turn()
    await turns.wait_for(1)

    assert turns.started[0]["turn_parts"][1] == {"type": "text", "text": "Answer in one word."}


def _bad_request() -> BadRequestError:
    response = httpx.Response(400, request=httpx.Request("POST", "http://localhost:8000/v1"))
    return BadRequestError("bad file", response=response, body=None)


class TestTurnErrors:
    def _service(self, error: Exception):
        service = _service()
        service.push_frame = AsyncMock()
        service.push_error = AsyncMock()
        service._call_event_handler = AsyncMock()
        service._process_context = AsyncMock(side_effect=error)
        context = LLMContext([{"role": "user", "content": "hi"}])
        context.remove_invalid_file_message = Mock(return_value=True)
        return service, context

    @staticmethod
    def _pushed_types(service) -> list[type]:
        return [type(call.args[0]) for call in service.push_frame.await_args_list]

    @pytest.mark.asyncio
    async def test_timeout_is_reported(self):
        service, context = self._service(TIMEOUT_EXCEPTIONS[0]("timed out"))
        await service._run_turn(context)

        service._call_event_handler.assert_awaited_with("on_completion_timeout")
        assert service.push_error.await_args.kwargs["error_msg"] == "LLM completion timeout"
        assert self._pushed_types(service) == [LLMFullResponseStartFrame, LLMFullResponseEndFrame]

    @pytest.mark.asyncio
    async def test_unconvertible_context_drops_the_invalid_file(self):
        service, context = self._service(LLMContextConversionError("bad media"))
        await service._run_turn(context)

        context.remove_invalid_file_message.assert_called_once()
        assert service.push_error.await_args.kwargs["error_msg"].endswith("bad media")

    @pytest.mark.asyncio
    async def test_rejected_file_is_an_application_error(self):
        service, context = self._service(_bad_request())
        await service._run_turn(context)

        context.remove_invalid_file_message.assert_called_once()
        assert service.push_error.await_args.kwargs["category"] == ErrorCategory.APPLICATION
        assert service._active_turn_parts is None

    @pytest.mark.asyncio
    async def test_audio_turn_forgets_the_previous_transcript(self):
        service, context = self._service(_bad_request())
        service._last_transcript = "an earlier question"
        await service._run_turn(context, expect_transcript=True)

        assert service._last_transcript == ""


class TestStreamHandling:
    def _service(self, *, expect_transcript: bool):
        service = _service()
        frames = []
        service.push_frame = AsyncMock(side_effect=lambda frame, *a, **kw: frames.append(frame))
        service._reset_response_state()
        service._transcript_extractor = (
            _TranscriptResponseExtractor() if expect_transcript else None
        )
        return service, frames

    async def _drain(self, service, *chunks):
        out = []
        async for chunk in service._handle_reasoning_content(_stream(*chunks)):
            out.append(chunk)
            delta = chunk.choices[0].delta
            if delta.content:
                await service._push_llm_text(delta.content)
        return out

    @staticmethod
    def _spoken(frames) -> str:
        return "".join(f.text for f in frames if isinstance(f, LLMTextFrame))

    @staticmethod
    def _transcripts(frames) -> list[str]:
        return [f.text for f in frames if isinstance(f, TranscriptionFrame)]

    @pytest.mark.asyncio
    async def test_transcript_is_split_from_the_spoken_reply(self):
        service, frames = self._service(expect_transcript=True)
        await self._drain(
            service,
            _chunk("<transcript>Where is the "),
            _chunk("Eiffel Tower?</transcript>"),
            _chunk("<response>It is in Paris."),
            _chunk("</response>"),
        )

        assert self._transcripts(frames) == ["Where is the Eiffel Tower?"]
        assert self._spoken(frames) == "It is in Paris."
        assert service._last_transcript == "Where is the Eiffel Tower?"

    @pytest.mark.asyncio
    async def test_untagged_reply_still_reaches_tts(self):
        service, frames = self._service(expect_transcript=True)
        await self._drain(service, _chunk("It is in Paris."))

        assert self._transcripts(frames) == []
        assert self._spoken(frames) == "It is in Paris."

    @pytest.mark.asyncio
    async def test_reply_held_back_at_stream_end_is_flushed(self):
        service, frames = self._service(expect_transcript=True)
        await self._drain(
            service, _chunk("<transcript>Hi</transcript><response>Hello the"), _chunk("re.")
        )

        assert self._spoken(frames) == "Hello there."

    @pytest.mark.asyncio
    async def test_think_tags_and_transcript_compose(self):
        service, frames = self._service(expect_transcript=True)
        await self._drain(
            service,
            _chunk("<think>They asked for a city.</think>"),
            _chunk("<transcript>Where is it?</transcript>"),
            _chunk("<response>In Paris.</response>"),
        )

        thought = "".join(f.text for f in frames if isinstance(f, LLMThoughtTextFrame))
        assert thought == "They asked for a city."
        assert isinstance(frames[0], LLMThoughtStartFrame)
        assert any(isinstance(f, LLMThoughtEndFrame) for f in frames)
        assert self._transcripts(frames) == ["Where is it?"]
        assert self._spoken(frames) == "In Paris."

    @pytest.mark.asyncio
    async def test_reasoning_content_stays_out_of_the_transcript(self):
        service, frames = self._service(expect_transcript=True)
        await self._drain(
            service,
            _chunk(reasoning_content="Thinking hard."),
            _chunk("<transcript>Hi</transcript><response>Hello.</response>"),
        )

        assert isinstance(frames[0], LLMThoughtStartFrame)
        assert frames[1].text == "Thinking hard."
        assert self._spoken(frames) == "Hello."

    @pytest.mark.asyncio
    async def test_transcript_cut_off_by_an_interruption_is_not_reported(self):
        service, frames = self._service(expect_transcript=True)
        await service._push_llm_text("<transcript>Where is the")
        await service._finalize_reasoning_state(flush_buffered_text=False)

        assert self._transcripts(frames) == []

    @pytest.mark.asyncio
    async def test_plain_text_is_not_rewritten(self):
        service, frames = self._service(expect_transcript=False)
        out = await self._drain(service, _chunk("Hello"), _chunk(" there."))

        assert [c.choices[0].delta.content for c in out] == ["Hello", " there."]
        assert self._spoken(frames) == "Hello there."


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return _stream(*self._chunks)

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_spoken_turn_is_transcribed_and_answered_in_a_pipeline():
    service = _service()
    create = AsyncMock(
        return_value=_FakeStream(
            [
                _chunk("<transcript>Where is the tower?</transcript>"),
                _chunk("<response>In Paris.</response>"),
            ]
        )
    )
    service._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    audio = b"\x00" * service._bytes_per_second()

    down_frames, up_frames = await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            UserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1),
            # The user aggregator ends the turn after VAD stops, and
            # LLMService tracks VAD speaking state separately from Omni.
            VADUserStoppedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.2),
        ],
    )

    spoken = "".join(f.text for f in down_frames if isinstance(f, LLMTextFrame))
    assert spoken == "In Paris."
    assert any(isinstance(f, LLMFullResponseStartFrame) for f in down_frames)
    assert any(isinstance(f, LLMFullResponseEndFrame) for f in down_frames)
    assert [f.text for f in up_frames if isinstance(f, TranscriptionFrame)] == [
        "Where is the tower?"
    ]
    request_parts = create.call_args.kwargs["messages"][-1]["content"]
    assert request_parts[0]["type"] == "audio_url"


@pytest.mark.asyncio
async def test_transcript_is_pushed_upstream_and_not_written_to_the_context():
    service = _service()
    context = LLMContext([{"role": "system", "content": "You are helpful."}])
    service._context = context
    pushed = []
    service.push_frame = AsyncMock(
        side_effect=lambda frame, direction=None: pushed.append((frame, direction))
    )
    extractor = _TranscriptResponseExtractor()
    extractor.feed("<transcript>where is the tower?</transcript>")

    await service._maybe_push_transcript(extractor)

    assert [m["role"] for m in context.get_messages()] == ["system"]
    frame, direction = pushed[-1]
    assert isinstance(frame, TranscriptionFrame)
    assert frame.text == "where is the tower?"
    assert direction == FrameDirection.UPSTREAM


def test_audio_input_announces_a_realtime_service():
    assert _service().service_metadata_frame().is_realtime_service
    assert not _service(audio_input=False).service_metadata_frame().is_realtime_service


@pytest.mark.asyncio
async def test_text_llm_ignores_user_audio():
    service = _service(audio_input=False)
    service.push_frame = AsyncMock()
    await _speak(service)

    assert service._pre_speech_buffer == []
    assert not service._turn_tasks


class _SlowStream:
    def __aiter__(self):
        async def chunks():
            await asyncio.sleep(0.2)
            yield _chunk("Goodbye!")

        return chunks()

    async def close(self):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("audio_input", [True, False])
async def test_reply_requested_before_the_end_frame_is_completed(audio_input):
    service = _service(audio_input=audio_input)
    service._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=_SlowStream()))
        )
    )

    down_frames, _ = await run_test(
        service,
        frames_to_send=[
            LLMContextFrame(LLMContext([{"role": "developer", "content": "Say goodbye."}]))
        ],
    )

    assert "".join(f.text for f in down_frames if isinstance(f, LLMTextFrame)) == "Goodbye!"


@pytest.mark.asyncio
async def test_current_turn_has_user_audio_only_during_an_audio_turn():
    service = _service()
    assert not service.current_turn_has_user_audio()
    service._turn_parts_var.set([_audio_part(b"\x00\x00", 16000, 1)])
    assert service.current_turn_has_user_audio()


def test_audio_turn_is_appended_without_changing_the_context_params():
    service = _service()
    params_from_context = {"messages": [{"role": "user", "content": "hi"}]}
    service._turn_parts_var.set([{"type": "text", "text": "listen"}])

    params = service.build_chat_completion_params(params_from_context)

    assert params["messages"][-1] == {
        "role": "user",
        "content": [{"type": "text", "text": "listen"}],
    }
    assert len(params_from_context["messages"]) == 1


def test_unset_token_limits_are_not_sent():
    params = _service().build_chat_completion_params({"messages": []})
    assert "max_tokens" not in params
    assert "max_completion_tokens" not in params


def _capture_requests(service: NvidiaOmniLLMService) -> dict:
    sent: dict = {}

    async def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    service._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return sent


@pytest.mark.asyncio
async def test_run_inference_sends_one_token_limit_and_no_audio_turn():
    service = _service(max_tokens=8192)
    service._turn_parts_var.set([{"type": "text", "text": "listen"}])
    sent = _capture_requests(service)

    await service.run_inference(LLMContext([{"role": "user", "content": "hi"}]), max_tokens=2048)

    assert sent["max_tokens"] == 2048
    assert "max_completion_tokens" not in sent
    assert sent["messages"] == [{"role": "user", "content": "hi"}]
    assert service._active_turn_parts == [{"type": "text", "text": "listen"}]


@pytest.mark.asyncio
async def test_developer_messages_are_sent_as_user_messages():
    service = _service()
    sent = _capture_requests(service)

    await service.run_inference(
        LLMContext([{"role": "developer", "content": "Introduce yourself."}])
    )

    assert sent["messages"] == [{"role": "user", "content": "Introduce yourself."}]


@pytest.mark.asyncio
async def test_turn_parts_are_only_sent_by_the_task_running_the_turn():
    service = _service()
    in_turn = asyncio.Event()
    release = asyncio.Event()

    async def turn():
        service._turn_parts_var.set([{"type": "text", "text": "listen"}])
        in_turn.set()
        await release.wait()
        return service.build_chat_completion_params({"messages": []})["messages"]

    turn_task = asyncio.create_task(turn())
    await in_turn.wait()
    concurrent = service.build_chat_completion_params({"messages": []})["messages"]
    release.set()

    assert concurrent == []
    assert await turn_task == [{"role": "user", "content": [{"type": "text", "text": "listen"}]}]


@pytest.mark.asyncio
async def test_multimodal_inference_returns_text_and_reasoning():
    service = _service(max_tokens=8192)
    sent: dict = {}

    async def create(**kwargs):
        sent.update(kwargs)
        message = SimpleNamespace(content=" Berlin. ", reasoning_content="Capital of Germany.")
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    service._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    result = await service.run_multimodal_inference(
        LLMContext([{"role": "user", "content": "Capital?"}]), max_tokens=256, reasoning_budget=64
    )

    assert result.text == "Berlin."
    assert result.reasoning == "Capital of Germany."
    assert result.finish_reason == "stop"
    assert sent["max_tokens"] == 256
    assert sent["extra_body"]["reasoning_budget"] == 64
    assert sent["stream"] is False


@pytest.mark.asyncio
async def test_multimodal_inference_reads_content_parts_and_extra_reasoning():
    service = _service()
    message = SimpleNamespace(
        content=[{"type": "text", "text": "Berlin."}, {"type": "image_url"}],
        model_extra={"reasoning": "Capital of Germany."},
    )

    async def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    service._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    result = await service.run_multimodal_inference(
        LLMContext([{"role": "user", "content": "Capital?"}])
    )

    assert (result.text, result.reasoning) == ("Berlin.", "Capital of Germany.")


@pytest.mark.asyncio
async def test_streamed_multimodal_inference_reports_deltas():
    service = _service()

    def chunk(content=None, reasoning=None, finish_reason=None):
        delta = SimpleNamespace(content=content, reasoning_content=reasoning)
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)])

    create = AsyncMock(
        return_value=_stream(
            SimpleNamespace(choices=[]),
            chunk(reasoning="Hmm."),
            chunk("Ber"),
            chunk("lin.", finish_reason="stop"),
        )
    )
    service._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    text_deltas, reasoning_deltas = [], []

    async def on_text(delta):
        text_deltas.append(delta)

    async def on_reasoning(delta):
        reasoning_deltas.append(delta)

    result = await service.run_multimodal_inference(
        LLMContext([{"role": "user", "content": "Capital?"}]),
        temperature=0.2,
        stream=True,
        on_text_delta=on_text,
        on_reasoning_delta=on_reasoning,
    )

    assert create.call_args.kwargs["temperature"] == 0.2
    assert (result.text, result.reasoning, result.finish_reason) == ("Berlin.", "Hmm.", "stop")
    assert text_deltas == ["Ber", "lin."]
    assert reasoning_deltas == ["Hmm."]


@pytest.mark.asyncio
async def test_text_only_service_works_as_a_plain_llm_in_a_pipeline():
    service = _service(audio_input=False)
    create = AsyncMock(return_value=_FakeStream([_chunk("Hello there.")]))
    service._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    audio = b"\x00" * service._bytes_per_second()

    down_frames, up_frames = await run_test(
        service,
        frames_to_send=[
            UserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1),
            UserStoppedSpeakingFrame(),
            LLMContextFrame(LLMContext([{"role": "user", "content": "hi"}])),
            SleepFrame(sleep=0.2),
        ],
    )

    assert "".join(f.text for f in down_frames if isinstance(f, LLMTextFrame)) == "Hello there."
    assert not any(isinstance(f, TranscriptionFrame) for f in up_frames)
    assert create.await_count == 1
    assert create.call_args.kwargs["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_subclass_can_report_transcripts_it_parses_itself():
    reported = []

    class Subclass(NvidiaOmniLLMService):
        async def _emit_user_transcript(self, transcript):
            reported.append(transcript)
            await super()._emit_user_transcript(transcript)

    service = Subclass(base_url="http://localhost:8000/v1")
    service.push_frame = AsyncMock()
    extractor = _TranscriptResponseExtractor()
    extractor.feed("<transcript>hello</transcript>")

    await service._maybe_push_transcript(extractor)

    assert reported == ["hello"]
    assert service._last_transcript == "hello"


class TestTranscriptResponseExtractor:
    @staticmethod
    def _feed(*chunks) -> tuple[_TranscriptResponseExtractor, str]:
        extractor = _TranscriptResponseExtractor()
        return extractor, "".join(extractor.feed(chunk) for chunk in chunks)

    def test_tags_split_across_chunks_and_whitespace(self):
        extractor, response = self._feed(
            "  ", "<trans", "cript>Hi</transcript>", "\n", "<resp", "onse>Hello.</response>", "x"
        )

        assert extractor.transcript == "Hi"
        assert response == "Hello."

    def test_untagged_text_after_the_transcript_is_the_response(self):
        extractor, response = self._feed("<transcript>Hi</transcript> Hello there.")

        assert extractor.transcript == "Hi"
        assert response == "Hello there."

    @pytest.mark.parametrize("tail", ["", "<", "</trans", "</transcript"])
    def test_unclosed_transcript_is_kept_at_stream_end(self, tail):
        extractor, response = self._feed("<transcript>Where is the tower", tail)

        assert extractor.finalize() == ""
        assert response == ""
        assert extractor.transcript_done
        assert extractor.transcript == "Where is the tower"

    def test_angle_bracket_in_an_unclosed_transcript_is_kept(self):
        extractor, _ = self._feed("<transcript>Is 3 < 5")
        extractor.finalize()

        assert extractor.transcript == "Is 3 < 5"


class TestAdapter:
    async def _converted_part(self, message):
        params = (
            await _service()
            .get_llm_adapter()
            .get_llm_invocation_params(LLMContext([message]), convert_developer_to_user=False)
        )
        return params["messages"][0]["content"][-1]

    @pytest.mark.asyncio
    async def test_input_audio_becomes_an_audio_url(self):
        message = {"role": "user", "content": [_audio_part(b"\x00\x00", 16000, 1)]}

        part = await self._converted_part(message)

        assert part["type"] == "audio_url"
        assert part["audio_url"]["url"].startswith("data:audio/wav;base64,")
        assert message["content"][0]["type"] == "input_audio"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("mime_type", "part_type"),
        [("audio/mpeg", "audio_url"), ("video/mp4", "video_url"), ("image/png", "image_url")],
    )
    async def test_media_files_use_the_part_omni_reads(self, mime_type, part_type):
        data_url = f"data:{mime_type};base64,{base64.b64encode(b'media').decode()}"
        message = await LLMContext.create_file_message(
            type="bytes", format=mime_type, file=data_url, text="Describe this."
        )

        part = await self._converted_part(message)

        assert part == {"type": part_type, part_type: {"url": data_url}}
