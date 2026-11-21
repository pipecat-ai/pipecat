#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for the Haru services.

``HaruLLMService`` is driven with scripted server events and recorded client
events. ``HaruLocalLLMService`` runs against a fake engine, so no test needs
the Haru packages, a checkpoint or a socket.
"""

import asyncio
import base64
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest

from pipecat.frames.frames import (
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    SpeechOutputAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.haru import local as haru_local
from pipecat.services.haru.base import HARU_FRAME_SAMPLES, HARU_SAMPLE_RATE
from pipecat.services.haru.llm import HaruLLMService, realtime_url
from pipecat.services.haru.local import HaruLocalLLMService, resolve_engine
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.base_object import BaseObject

#: Turn gap short enough to keep tests quick, long enough to survive scheduling.
TEST_TURN_GAP_SECS = 0.05

#: Bytes in one Haru frame as 16-bit PCM.
FRAME_BYTES = HARU_FRAME_SAMPLES * 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeWebSocket:
    """Minimal async-iterable websocket that yields scripted JSON strings."""

    def __init__(self, messages: list[str]):
        self._messages = list(messages)

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)

    async def close(self) -> None:
        self._messages.clear()


class _FrameRecorder:
    """Records frames passed to a service's ``push_frame``."""

    def __init__(self):
        self.frames: list[tuple[Any, FrameDirection]] = []

    async def __call__(self, frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        self.frames.append((frame, direction))

    def of_types(self, *types) -> list[Any]:
        return [f for f, _ in self.frames if isinstance(f, types)]


class _EventRecorder:
    """Records the client events a service sends."""

    def __init__(self):
        self.events: list[dict[str, Any]] = []

    async def __call__(self, event: dict[str, Any]):
        self.events.append(event)

    def of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == event_type]


async def _remote(**kwargs) -> tuple[HaruLLMService, _EventRecorder, _FrameRecorder]:
    """A remote service with its socket and its pipeline replaced by recorders."""
    service = HaruLLMService(base_url="http://haru.test:8000", **kwargs)
    service._assistant_turn.gap_secs = TEST_TURN_GAP_SECS
    await BaseObject.setup(service, TaskManager())
    sent, pushed = _EventRecorder(), _FrameRecorder()
    service._send_event = sent
    service.push_frame = pushed
    return service, sent, pushed


async def _drive(service: HaruLLMService, scripted: list[dict[str, Any]]) -> None:
    """Feed scripted server events through the receive handler."""
    service._websocket = _FakeWebSocket([json.dumps(e) for e in scripted])
    await service._receive_task_handler()


def _context() -> LLMContext:
    return LLMContext()


def _pcm16(samples: int, value: int = 1000) -> bytes:
    return np.full(samples, value, dtype="<i2").tobytes()


def _decode_f32(event: dict[str, Any]) -> np.ndarray:
    return np.frombuffer(base64.b64decode(event["audio"]), dtype="<f4")


# ---------------------------------------------------------------------------
# HaruLLMService: endpoint URL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_url, expected",
    [
        ("http://localhost:8000", "ws://localhost:8000/v1/realtime?duplex=1"),
        (
            "https://x.endpoints.huggingface.cloud/",
            "wss://x.endpoints.huggingface.cloud/v1/realtime?duplex=1",
        ),
        ("ws://gpu:8000/v1/realtime", "ws://gpu:8000/v1/realtime?duplex=1"),
        ("wss://gpu/v1/realtime?duplex=1", "wss://gpu/v1/realtime?duplex=1"),
        ("wss://gpu/proxy/v1/realtime?token=abc", "wss://gpu/proxy/v1/realtime?token=abc&duplex=1"),
    ],
)
def test_realtime_url(base_url, expected):
    """Any form of the server's address resolves to its duplex realtime endpoint."""
    assert realtime_url(base_url) == expected


# ---------------------------------------------------------------------------
# HaruLLMService: session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_update_carries_the_card_voice_and_options(tmp_path):
    """The first context opens the session with the card, the voice and the sampling options."""
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"RIFFfake")
    service, sent, _ = await _remote(
        voice_reference=voice,
        settings=HaruLLMService.Settings(
            system_instruction="You are Haru, a nurse in Denver.",
            temperature=0.5,
            guidance_scale=1.2,
            speak_first=True,
        ),
    )

    await service._handle_context(_context())

    (update,) = sent.of_type("session.update")
    session = update["session"]
    assert session["instructions"] == "You are Haru, a nurse in Denver."
    assert session["input_audio_format"] == "pcm_f32le"
    assert session["output_audio_format"] == "pcm16"
    assert session["sample_rate_hz"] == HARU_SAMPLE_RATE
    assert session["turn_detection"] is None
    assert session["temperature"] == 0.5
    assert session["extra_body"]["guidance_scale"] == 1.2
    assert session["extra_body"]["speak_first"] is True
    # Options left at None are the server's to default.
    assert "top_k" not in session["extra_body"]
    assert session["ref_audio"].startswith("data:audio/")
    assert base64.b64decode(session["ref_audio"].split(";base64,")[1]) == b"RIFFfake"


@pytest.mark.asyncio
async def test_session_without_a_voice_sends_no_reference():
    """With no clip configured the session carries no ``ref_audio``."""
    service, sent, _ = await _remote()
    await service._handle_context(_context())
    session = sent.of_type("session.update")[0]["session"]
    assert "ref_audio" not in session
    assert session["instructions"] == ""


@pytest.mark.asyncio
async def test_only_first_context_opens_a_session():
    """Later contexts carry what the aggregators recorded; the session is not reopened."""
    service, sent, _ = await _remote()
    await service._handle_context(_context())
    await service._handle_context(_context())
    assert len(sent.of_type("session.update")) == 1


@pytest.mark.asyncio
async def test_session_created_starts_the_conversation():
    """``session.created`` is answered with one silent frame, which makes the model run its prompt."""
    service, sent, pushed = await _remote()
    started = AsyncMock()
    service.add_event_handler("on_session_started", started)
    await service._handle_context(_context())

    await _drive(service, [{"type": "session.created", "session": {"id": "s1"}}])

    (append,) = sent.of_type("input_audio_buffer.append")
    assert append["format"] == "pcm_f32le"
    assert append["sample_rate_hz"] == HARU_SAMPLE_RATE
    silence = _decode_f32(append)
    assert silence.shape == (HARU_FRAME_SAMPLES,)
    assert not silence.any()
    # The prompt is still running: user audio sent now would only queue up behind it.
    assert not service._session_started

    frame = {"type": "response.output_audio.delta", "delta": base64.b64encode(_pcm16(8)).decode()}
    await _drive(service, [frame, frame])
    await asyncio.sleep(0.01)

    assert service._session_started
    started.assert_awaited_once()
    assert len(pushed.of_types(SpeechOutputAudioRawFrame)) == 2


# ---------------------------------------------------------------------------
# HaruLLMService: audio
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_input_audio_dropped_until_session_started():
    """Audio that arrives before the model is listening is not buffered for later."""
    service, sent, _ = await _remote()
    frame = InputAudioRawFrame(
        audio=_pcm16(HARU_FRAME_SAMPLES), sample_rate=HARU_SAMPLE_RATE, num_channels=1
    )
    await service.process_frame(frame, FrameDirection.DOWNSTREAM)
    assert not sent.of_type("input_audio_buffer.append")


@pytest.mark.asyncio
async def test_input_audio_is_sent_in_whole_frames():
    """Transport chunks are regrouped into the 3200-sample frames the server requires."""
    service, sent, _ = await _remote()
    service._session_started = True

    # 20 ms chunks at the model's rate: 480 samples each, 3200 per model frame.
    chunk = InputAudioRawFrame(audio=_pcm16(480), sample_rate=HARU_SAMPLE_RATE, num_channels=1)
    for _ in range(14):  # 6720 samples: two frames and a remainder
        await service.process_frame(chunk, FrameDirection.DOWNSTREAM)

    appends = sent.of_type("input_audio_buffer.append")
    assert len(appends) == 2
    for append in appends:
        samples = _decode_f32(append)
        assert samples.shape == (HARU_FRAME_SAMPLES,)
        assert samples[0] == pytest.approx(1000 / 32768)


@pytest.mark.asyncio
async def test_input_audio_is_resampled_to_the_model_rate():
    """Audio at another rate reaches the server at 24 kHz."""
    service, sent, _ = await _remote()
    service._session_started = True

    one_second = InputAudioRawFrame(audio=_pcm16(16000), sample_rate=16000, num_channels=1)
    await service.process_frame(one_second, FrameDirection.DOWNSTREAM)

    # One second is 7.5 frames; the resampler may hold back a little of its tail.
    assert len(sent.of_type("input_audio_buffer.append")) in (6, 7)


@pytest.mark.asyncio
async def test_output_audio_is_pushed_as_a_speech_stream():
    """The model's audio is continuous, so it is a speech stream rather than TTS output."""
    service, _, pushed = await _remote()
    pcm = _pcm16(HARU_FRAME_SAMPLES, value=7)
    await _drive(
        service,
        [{"type": "response.output_audio.delta", "delta": base64.b64encode(pcm).decode()}],
    )

    (frame,) = pushed.of_types(SpeechOutputAudioRawFrame)
    assert frame.audio == pcm
    assert frame.sample_rate == HARU_SAMPLE_RATE
    assert frame.num_channels == 1


# ---------------------------------------------------------------------------
# HaruLLMService: text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transcript_is_grouped_into_assistant_turns():
    """Text arrives per frame; a quiet gap closes the turn it belongs to."""
    service, _, pushed = await _remote()
    await _drive(
        service,
        [
            {"type": "response.output_audio_transcript.delta", "delta": "Hello"},
            {"type": "response.output_audio_transcript.delta", "delta": " there"},
        ],
    )
    await asyncio.sleep(TEST_TURN_GAP_SECS * 3)

    kinds = [
        type(f)
        for f, _ in pushed.frames
        if isinstance(
            f,
            (
                LLMFullResponseStartFrame,
                TTSStartedFrame,
                TTSTextFrame,
                TTSStoppedFrame,
                LLMFullResponseEndFrame,
            ),
        )
    ]
    assert kinds == [
        LLMFullResponseStartFrame,
        TTSStartedFrame,
        TTSTextFrame,
        TTSTextFrame,
        TTSStoppedFrame,
        LLMFullResponseEndFrame,
    ]
    assert [f.text for f in pushed.of_types(TTSTextFrame)] == ["Hello", " there"]
    # The aggregator records the TTSTextFrame, so the LLMTextFrame must not be recorded too.
    assert all(f.append_to_context is False for f in pushed.of_types(LLMTextFrame))


@pytest.mark.asyncio
async def test_whitespace_alone_does_not_open_a_turn():
    """A frame of only whitespace would otherwise be recorded as an empty assistant message."""
    service, _, pushed = await _remote()
    await _drive(service, [{"type": "response.output_audio_transcript.delta", "delta": " "}])
    await asyncio.sleep(TEST_TURN_GAP_SECS * 3)
    assert not pushed.of_types(LLMFullResponseStartFrame, TTSTextFrame)


# ---------------------------------------------------------------------------
# HaruLLMService: errors and pipeline integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_before_session_is_permanent():
    """An error before ``session.created`` means this session will never start."""
    service, _, _ = await _remote()
    service.push_error = AsyncMock()
    await _drive(service, [{"type": "error", "error": {"message": "invalid_ref_audio"}}])

    assert "invalid_ref_audio" in service.push_error.await_args.kwargs["error_msg"]
    assert service.push_error.await_args.kwargs.get("force_treat_as_permanent") is True


@pytest.mark.asyncio
async def test_error_after_session_is_not_permanent():
    """Once the conversation is running, an error is reported without ending it."""
    service, _, _ = await _remote()
    service._session_started = True
    service.push_error = AsyncMock()
    await _drive(service, [{"type": "error", "error": "frame rejected"}])

    assert service.push_error.await_args.kwargs.get("force_treat_as_permanent") is not True


def test_metadata_recommends_no_interruptions():
    """The model handles being talked over itself, so the pipeline must not interrupt it."""
    service = HaruLLMService(base_url="http://haru.test:8000")
    strategies = service.service_metadata_frame().user_turn_strategies
    assert isinstance(strategies, ExternalUserTurnStrategies)
    assert strategies.enable_interruptions is False


def test_api_key_becomes_a_bearer_header():
    """A hosted endpoint is authorized with a bearer token; a bare server needs none."""
    assert HaruLLMService(base_url="http://h")._headers() == {}
    assert HaruLLMService(base_url="http://h", api_key="hf_x")._headers() == {
        "Authorization": "Bearer hf_x"
    }


# ---------------------------------------------------------------------------
# HaruLocalLLMService
# ---------------------------------------------------------------------------


class _FakeSession:
    """Stands in for a Haru ``DuplexSession``: echoes a counter and scripted text."""

    def __init__(self, model, system_prompt, reference, config):
        self.model = model
        self.system_prompt = system_prompt
        self.reference = reference
        self.config = config
        self.steps: list[np.ndarray] = []
        self.texts: dict[int, str] = {}
        self.index = 0

    def _frame(self):
        frame = SimpleNamespace(
            audio=np.full(HARU_FRAME_SAMPLES, self.index / 100, dtype=np.float32),
            text=self.texts.get(self.index, ""),
            index=self.index,
            seconds=0.01,
            done=False,
        )
        self.index += 1
        return frame

    def start(self):
        return self._frame()

    def step(self, user_audio):
        self.steps.append(np.array(user_audio))
        return self._frame()


def _fake_engine() -> SimpleNamespace:
    """The parts of the ``haru`` / ``haru_mlx`` packages the service uses."""
    return SimpleNamespace(
        name="fake",
        model=object(),
        DuplexSession=_FakeSession,
        DuplexConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        load_reference=lambda path: np.ones(10, dtype=np.float32),
    )


async def _local(monkeypatch, **kwargs) -> tuple[HaruLocalLLMService, _FrameRecorder]:
    monkeypatch.setattr(haru_local, "load_engine", lambda *a, **k: _fake_engine())
    service = HaruLocalLLMService(**kwargs)
    service._assistant_turn.gap_secs = TEST_TURN_GAP_SECS
    await BaseObject.setup(service, TaskManager())
    pushed = _FrameRecorder()
    service.push_frame = pushed
    return service, pushed


async def _until(condition, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "timed out"
        await asyncio.sleep(0.005)


@pytest.mark.parametrize(
    "requested, system, machine, expected",
    [
        ("auto", "Darwin", "arm64", "mlx"),
        ("auto", "Linux", "x86_64", "torch"),
        ("auto", "Darwin", "x86_64", "torch"),
        ("torch", "Darwin", "arm64", "torch"),
        ("mlx", "Darwin", "arm64", "mlx"),
    ],
)
def test_resolve_engine(requested, system, machine, expected):
    """``auto`` picks MLX on Apple silicon and PyTorch everywhere else."""
    assert resolve_engine(requested, system=system, machine=machine) == expected


def test_resolve_engine_rejects_unknown():
    with pytest.raises(ValueError, match="engine"):
        resolve_engine("onnx", system="Linux", machine="x86_64")


@pytest.mark.asyncio
async def test_local_session_runs_the_prompt_and_speaks_frame_zero(monkeypatch, tmp_path):
    """The first context builds the session and plays the model's first frame."""
    service, pushed = await _local(
        monkeypatch,
        voice_reference=tmp_path / "voice.wav",
        settings=HaruLocalLLMService.Settings(
            system_instruction="You are Haru.", temperature=0.4, top_k=10, speak_first=True, seed=3
        ),
    )
    started = AsyncMock()
    service.add_event_handler("on_session_started", started)

    await service._handle_context(_context())
    await _until(lambda: service._session_started)

    session = service._session
    assert session.system_prompt == "You are Haru."
    assert session.reference is not None
    assert session.config.temperature == 0.4
    assert session.config.top_k == 10
    assert session.config.speak_first is True
    assert session.config.seed == 3
    # Options left at None keep the engine's own defaults.
    assert not hasattr(session.config, "guidance_scale")
    assert len(pushed.of_types(SpeechOutputAudioRawFrame)) == 1
    started.assert_awaited_once()
    await service.cleanup()


@pytest.mark.asyncio
async def test_local_steps_one_model_frame_per_user_frame(monkeypatch):
    """Each whole user frame is stepped in order and answered by one frame of audio and text."""
    service, pushed = await _local(monkeypatch)
    await service._handle_context(_context())
    await _until(lambda: service._session_started)
    service._session.texts = {1: "Hi", 2: " Sam"}

    audio = InputAudioRawFrame(
        audio=_pcm16(HARU_FRAME_SAMPLES * 3, value=16384),
        sample_rate=HARU_SAMPLE_RATE,
        num_channels=1,
    )
    await service.process_frame(audio, FrameDirection.DOWNSTREAM)
    await _until(lambda: len(service._session.steps) == 3)
    await _until(lambda: len(pushed.of_types(SpeechOutputAudioRawFrame)) == 4)

    step = service._session.steps[0]
    assert step.dtype == np.float32
    assert step.shape == (HARU_FRAME_SAMPLES,)
    assert step[0] == pytest.approx(0.5)

    frames = pushed.of_types(SpeechOutputAudioRawFrame)
    assert all(len(f.audio) == FRAME_BYTES for f in frames)
    # Frame k carries k/100 as its sample value, so the order is checkable.
    values = [int(np.frombuffer(f.audio, dtype="<i2")[0]) for f in frames]
    assert values == sorted(values)
    assert [f.text for f in pushed.of_types(TTSTextFrame)] == ["Hi", " Sam"]
    await service.cleanup()


@pytest.mark.asyncio
async def test_local_drops_oldest_audio_when_the_engine_falls_behind(monkeypatch):
    """A backlog would only add latency, so the oldest waiting frames are dropped."""
    service, _ = await _local(monkeypatch, max_backlog_frames=2)
    service._session_started = True  # no step task: nothing drains the queue

    audio = InputAudioRawFrame(
        audio=b"".join(_pcm16(HARU_FRAME_SAMPLES, value=v) for v in (100, 200, 300, 400)),
        sample_rate=HARU_SAMPLE_RATE,
        num_channels=1,
    )
    await service.process_frame(audio, FrameDirection.DOWNSTREAM)

    waiting = [service._user_frames.get_nowait() for _ in range(service._user_frames.qsize())]
    assert [round(float(f[0]) * 32768) for f in waiting] == [300, 400]
    await service.cleanup()


def test_local_missing_engine_package_names_the_extra(monkeypatch):
    """Without the engine's package installed, the error says which extra provides it."""
    monkeypatch.setattr(haru_local, "resolve_engine", lambda *a, **k: "mlx")
    monkeypatch.setitem(__import__("sys").modules, "haru_mlx", None)
    with pytest.raises(ImportError, match=r"pipecat-ai\[haru-mlx\]"):
        haru_local.load_engine("mlx", "higurashi-ai/Haru-4B")
