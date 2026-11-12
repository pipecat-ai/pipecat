#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Meta streaming STT service handshake and event handling."""

import json
from unittest.mock import AsyncMock

import pytest
from websockets.protocol import State

from pipecat import version as pipecat_version
from pipecat.frames.frames import InputAudioRawFrame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.meta.stt import MetaSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.errors import ErrorCategory


class _FakeWebsocket:
    """A websocket that replays server messages and records what was sent."""

    def __init__(self, messages=None, *, ack=None, state=State.OPEN):
        self._messages = messages or []
        self._ack = json.dumps({"sessionId": "stream-123"} if ack is None else ack)
        self.state = state
        self.sent = []
        self.closed = False
        self.connect_kwargs = {}

    async def send(self, payload):
        self.sent.append(payload)

    async def recv(self):
        return self._ack

    async def close(self):
        self.closed = True
        self.state = State.CLOSED

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for message in self._messages:
            yield message


def _patch_connect(monkeypatch, websocket: _FakeWebsocket) -> _FakeWebsocket:
    async def fake_websocket_connect(*args, **kwargs):
        websocket.connect_kwargs = kwargs
        return websocket

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect", fake_websocket_connect
    )
    return websocket


def _service(sample_rate: int = 16000, **kwargs) -> MetaSTTService:
    service = MetaSTTService(api_key="test-key", **kwargs)
    # sample_rate is normally set from StartFrame, which these tests skip.
    service._sample_rate = sample_rate
    return service


def _handshake(websocket: _FakeWebsocket) -> dict:
    return json.loads(websocket.sent[0])


async def _connected(monkeypatch, sample_rate: int = 16000, **kwargs):
    websocket = _patch_connect(monkeypatch, _FakeWebsocket())
    service = _service(sample_rate, **kwargs)
    await service._connect_websocket()
    return service, websocket


@pytest.mark.asyncio
async def test_handshake_carries_credential_and_session_config(monkeypatch):
    service, websocket = await _connected(monkeypatch)

    handshake = _handshake(websocket)
    assert handshake["authorization"]["accessToken"] == "Bearer test-key"
    assert handshake["model"] == "muse-voice-transcribe-1.0"
    assert handshake["mode"] == "ENDPOINTING"
    assert handshake["partialMode"] == "CUMULATIVE"
    assert handshake["emitAudioProgress"] is False
    assert handshake["languageBias"] == ["English"]


@pytest.mark.asyncio
async def test_connect_identifies_the_pipecat_integration(monkeypatch):
    _, websocket = await _connected(monkeypatch)

    user_agent = websocket.connect_kwargs["additional_headers"]["User-Agent"]
    assert user_agent == f"Meta/1.0 (integration=Pipecat/{pipecat_version()})"


@pytest.mark.asyncio
async def test_handshake_language_bias_replaces_the_single_language(monkeypatch):
    _, websocket = await _connected(
        monkeypatch,
        settings=MetaSTTService.Settings(language_bias=[Language.EN, Language.FR]),
    )

    assert _handshake(websocket)["languageBias"] == ["English", "French"]


@pytest.mark.asyncio
async def test_handshake_omits_language_bias_when_language_is_unset(monkeypatch):
    _, websocket = await _connected(monkeypatch, settings=MetaSTTService.Settings(language=None))

    assert "languageBias" not in _handshake(websocket)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sample_rate, encoding",
    [(16000, "PCM_16KHZ"), (24000, "PCM_24KHZ"), (8000, "PCM_16KHZ")],
)
async def test_audio_encoding_follows_the_pipeline_sample_rate(monkeypatch, sample_rate, encoding):
    _, websocket = await _connected(monkeypatch, sample_rate)

    assert _handshake(websocket)["audioEncoding"] == encoding


@pytest.mark.asyncio
async def test_off_rate_audio_is_resampled_before_it_is_sent(monkeypatch):
    service, websocket = await _connected(monkeypatch, 8000)

    for _ in range(10):
        frame = InputAudioRawFrame(audio=b"\x00" * 1600, sample_rate=8000, num_channels=1)
        await service.process_audio_frame(frame, FrameDirection.DOWNSTREAM)

    audio_sent = b"".join(chunk for chunk in websocket.sent if isinstance(chunk, bytes))
    # 8 kHz in, 16 kHz out: twice the bytes, less what the resampler still holds.
    assert 0.9 < len(audio_sent) / (2 * 10 * 1600) <= 1.0


@pytest.mark.asyncio
async def test_native_rate_audio_is_sent_unchanged(monkeypatch):
    service, websocket = await _connected(monkeypatch, 24000)

    async for _ in service.run_stt(b"\x01" * 480):
        pass

    assert websocket.sent[-1] == b"\x01" * 480


@pytest.mark.asyncio
async def test_accepted_handshake_readies_the_session(monkeypatch):
    service, _ = await _connected(monkeypatch)

    assert service._session_ready.is_set()
    assert service._session_id == "stream-123"


@pytest.mark.asyncio
async def test_rejected_handshake_makes_the_service_unusable(monkeypatch):
    websocket = _patch_connect(
        monkeypatch, _FakeWebsocket(ack={"type": "error", "message": "invalid credential"})
    )
    service = _service()
    errors = []
    service.push_frame = AsyncMock(side_effect=lambda frame, *a, **kw: errors.append(frame))

    await service._connect_websocket()

    assert service._websocket is None
    assert websocket.closed
    assert not service.is_usable
    assert errors[0].category == ErrorCategory.AUTHENTICATION


@pytest.mark.asyncio
async def test_endpointing_turn_emits_interims_then_the_completed_transcript(monkeypatch):
    service = _service()
    service._websocket = _FakeWebsocket(
        [
            json.dumps({"type": "speechStart", "turnId": 1, "audioProcessedMs": 100}),
            json.dumps({"type": "transcript", "transcript": "what is the", "final": False}),
            json.dumps({"type": "transcript", "transcript": "what is the capital", "final": False}),
            json.dumps({"type": "speechEnd", "turnId": 1, "audioProcessedMs": 900}),
            json.dumps(
                {
                    "type": "speechComplete",
                    "turnId": 1,
                    "transcript": "What is the capital?",
                    "audioProcessedMs": 900,
                }
            ),
        ]
    )
    frames = []
    monkeypatch.setattr(service, "push_frame", AsyncMock(side_effect=lambda f: frames.append(f)))
    monkeypatch.setattr(service, "emit_stt_usage_metrics", AsyncMock())

    await service._receive_messages()

    assert [type(frame) for frame in frames] == [
        InterimTranscriptionFrame,
        InterimTranscriptionFrame,
        TranscriptionFrame,
    ]
    assert frames[-1].text == "What is the capital?"
    assert frames[-1].finalized
    assert frames[-1].language is Language.EN


@pytest.mark.asyncio
async def test_final_transcript_is_interim_until_the_turn_completes(monkeypatch):
    service = _service()
    service._websocket = _FakeWebsocket(
        [json.dumps({"type": "transcript", "transcript": "Berlin.", "final": True})]
    )
    frames = []
    monkeypatch.setattr(service, "push_frame", AsyncMock(side_effect=lambda f: frames.append(f)))

    await service._receive_messages()

    assert [type(frame) for frame in frames] == [InterimTranscriptionFrame]


@pytest.mark.asyncio
async def test_final_transcript_ends_the_turn_in_push_to_talk(monkeypatch):
    service = _service(settings=MetaSTTService.Settings(mode="PUSH_TO_TALK"))
    service._websocket = _FakeWebsocket(
        [json.dumps({"type": "transcript", "transcript": "Berlin.", "final": True})]
    )
    frames = []
    monkeypatch.setattr(service, "push_frame", AsyncMock(side_effect=lambda f: frames.append(f)))
    monkeypatch.setattr(service, "emit_stt_usage_metrics", AsyncMock())

    await service._receive_messages()

    assert [type(frame) for frame in frames] == [TranscriptionFrame]


@pytest.mark.asyncio
async def test_usage_is_reported_before_the_transcription_frame(monkeypatch):
    service = _service()
    service._websocket = _FakeWebsocket(
        [json.dumps({"type": "speechComplete", "turnId": 1, "transcript": "Berlin."})]
    )
    events = []
    monkeypatch.setattr(
        service, "push_frame", AsyncMock(side_effect=lambda f: events.append(type(f).__name__))
    )
    monkeypatch.setattr(
        service, "emit_stt_usage_metrics", AsyncMock(side_effect=lambda: events.append("usage"))
    )

    await service._receive_messages()

    assert events == ["usage", "TranscriptionFrame"]


@pytest.mark.asyncio
async def test_error_event_is_reported(monkeypatch):
    service = _service()
    service._websocket = _FakeWebsocket(
        [json.dumps({"type": "error", "message": "backend failure", "sessionId": "stream-123"})]
    )
    push_error = AsyncMock()
    monkeypatch.setattr(service, "push_error", push_error)

    await service._receive_messages()

    assert "backend failure" in push_error.await_args.kwargs["error_msg"]


@pytest.mark.asyncio
async def test_settings_change_reconnects(monkeypatch):
    service = _service()
    reconnect = AsyncMock()
    monkeypatch.setattr(service, "_request_reconnect", reconnect)

    await service._update_settings(MetaSTTService.Settings(keywords=["Pipecat"]))

    assert service._settings.keywords == ["Pipecat"]
    reconnect.assert_awaited_once()


def _diarized_turn(turn_id: int, label: str, transcript: str) -> list[str]:
    """A turn as Meta sends it in DIARIZATION mode, speaker event before speechEnd."""
    return [
        json.dumps({"type": "speechStart", "turnId": turn_id, "audioProcessedMs": 0}),
        json.dumps({"type": "transcript", "transcript": transcript, "final": False}),
        json.dumps({"type": "speaker", "label": label, "audioProcessedMs": 900}),
        json.dumps({"type": "speechEnd", "turnId": turn_id, "audioProcessedMs": 800}),
        json.dumps({"type": "speechComplete", "turnId": turn_id, "transcript": transcript}),
    ]


def _final_frames(monkeypatch, service) -> list[TranscriptionFrame]:
    frames = []
    monkeypatch.setattr(service, "push_frame", AsyncMock(side_effect=lambda f: frames.append(f)))
    monkeypatch.setattr(service, "emit_stt_usage_metrics", AsyncMock())
    return frames


@pytest.mark.asyncio
async def test_diarized_turns_carry_their_speaker(monkeypatch):
    service = _service(
        settings=MetaSTTService.Settings(
            mode="DIARIZATION", speaker_format="[Speaker {speaker_number}] {text}"
        )
    )
    service._websocket = _FakeWebsocket(
        _diarized_turn(0, "A", "My favorite fruit is the mango.")
        + _diarized_turn(1, "B", "Mine is the banana.")
        + _diarized_turn(2, "A", "I take the train.")
    )
    frames = _final_frames(monkeypatch, service)

    await service._receive_messages()

    finals = [f for f in frames if isinstance(f, TranscriptionFrame)]
    assert [(f.speaker.label, f.speaker.number, f.text) for f in finals] == [
        ("A", 1, "[Speaker 1] My favorite fruit is the mango."),
        ("B", 2, "[Speaker 2] Mine is the banana."),
        ("A", 1, "[Speaker 1] I take the train."),
    ]
    assert all(f.speaker is None for f in frames if isinstance(f, InterimTranscriptionFrame))


@pytest.mark.asyncio
async def test_speaker_belongs_to_the_turn_open_when_it_arrives(monkeypatch):
    service = _service(settings=MetaSTTService.Settings(mode="DIARIZATION"))
    service._websocket = _FakeWebsocket(
        [
            json.dumps({"type": "speechStart", "turnId": 0}),
            json.dumps({"type": "speaker", "label": "A"}),
            json.dumps({"type": "speechEnd", "turnId": 0}),
            # The next turn opens before the first one completes.
            json.dumps({"type": "speechStart", "turnId": 1}),
            json.dumps({"type": "speechComplete", "turnId": 0, "transcript": "Hello."}),
            json.dumps({"type": "speaker", "label": "B"}),
            json.dumps({"type": "speechComplete", "turnId": 1, "transcript": "Hi."}),
        ]
    )
    frames = _final_frames(monkeypatch, service)

    await service._receive_messages()

    assert [(f.speaker.label, f.text) for f in frames] == [("A", "Hello."), ("B", "Hi.")]


@pytest.mark.asyncio
async def test_turn_with_several_speakers_is_not_labelled(monkeypatch):
    service = _service(settings=MetaSTTService.Settings(mode="DIARIZATION"))
    service._websocket = _FakeWebsocket(
        [
            json.dumps({"type": "speechStart", "turnId": 0}),
            json.dumps({"type": "speaker", "label": "A"}),
            json.dumps({"type": "speaker", "label": "B"}),
            json.dumps({"type": "speechComplete", "turnId": 0, "transcript": "Hello. Hi."}),
        ]
    )
    frames = _final_frames(monkeypatch, service)

    await service._receive_messages()

    [frame] = frames
    assert frame.speaker is None
    assert frame.text == "Hello. Hi."


@pytest.mark.asyncio
async def test_speaker_format_change_does_not_reconnect(monkeypatch):
    service = _service()
    reconnect = AsyncMock()
    monkeypatch.setattr(service, "_request_reconnect", reconnect)

    changed = await service._update_settings(
        MetaSTTService.Settings(speaker_format="{speaker}: {text}")
    )

    assert "speaker_format" in changed
    reconnect.assert_not_awaited()
    assert "speaker_format" not in json.dumps(service._build_handshake())


@pytest.mark.asyncio
async def test_disconnect_half_closes_the_input_stream(monkeypatch):
    service, websocket = await _connected(monkeypatch)

    await service._disconnect()

    assert json.loads(websocket.sent[-1]) == {"type": "endStream"}
    assert websocket.closed
