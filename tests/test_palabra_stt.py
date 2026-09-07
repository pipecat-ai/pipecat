import json
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import pytest
from websockets.protocol import State

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TranslationFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.services.palabra.stt import (
    FINALIZE_MESSAGE,
    PalabraSTTService,
)
from pipecat.transcriptions.language import Language


class _FakeWebsocket:
    def __init__(self, messages=(), *, state=State.OPEN):
        self._messages = list(messages)
        self.state = state
        self.sent = []

    async def send(self, payload):
        self.sent.append(payload)

    async def close(self):
        self.state = State.CLOSED

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for message in self._messages:
            yield message


def _service(monkeypatch=None, messages=(), **kwargs) -> tuple[PalabraSTTService, list]:
    service = PalabraSTTService(api_key="test-key", **kwargs)
    service._sample_rate = 16000
    service._websocket = _FakeWebsocket(messages)

    pushed = []

    async def fake_push_frame(frame, direction=FrameDirection.DOWNSTREAM):
        await service._update_transcription_metrics(frame)
        pushed.append(frame)

    async def fake_noop(*args, **kwargs):
        pass

    if monkeypatch:
        monkeypatch.setattr(service, "push_frame", fake_push_frame)
        monkeypatch.setattr(service, "emit_stt_usage_metrics", fake_noop)
        monkeypatch.setattr(service, "push_error", fake_noop)
    return service, pushed


def _query(url: str) -> dict[str, str]:
    return {key: value[0] for key, value in parse_qs(urlparse(url).query).items()}


def _transcription(
    text: str, *, is_eos: bool, language: str = "en", segment: str | None = None
) -> str:
    """Build a Palabra transcription message whose delta carries ``text``."""
    return json.dumps(
        {
            "message_type": "transcription",
            "transcription_id": "segment-1",
            "language": language,
            "is_eos": is_eos,
            "delta": {"text": text},
            "segment": {
                "text": text if segment is None else segment,
                "start_time": 0.0,
                "end_time": 0.8,
            },
        }
    )


def test_url_carries_required_parameters_without_warm_language():
    service, _ = _service()

    url = service._build_url()

    assert url.startswith("wss://stream.palabra.ai/asr/v1/speech-to-text/stream?")
    assert _query(url) == {
        "token": "test-key",
        "format": "pcm_s16le",
        "sample_rate": "16000",
    }


def test_url_never_sends_warm_language():
    service, _ = _service(settings=PalabraSTTService.Settings(language=Language.EN))

    assert "language" not in _query(service._build_url())


def test_url_carries_translate_languages_lowercased_and_deduplicated():
    service, _ = _service(
        settings=PalabraSTTService.Settings(
            translate_languages=[Language.ES, Language.FR_CA, Language.ES]
        )
    )

    assert _query(service._build_url())["translate_languages"] == "es,fr-ca"


def test_url_carries_optional_settings():
    service, _ = _service(
        settings=PalabraSTTService.Settings(
            enable_filler_filter=False,
            finalization_mode="manual",
            finalization_timeout=30,
        )
    )

    query = _query(service._build_url())

    assert query["enable_filler_filter"] == "false"
    assert query["finalization_mode"] == "manual"
    assert query["finalization_timeout"] == "30"


@pytest.mark.asyncio
async def test_each_audio_frame_is_sent_immediately():
    service, _ = _service()

    async for _ in service.run_stt(b"first"):
        pass
    async for _ in service.run_stt(b"second"):
        pass

    assert service._websocket.sent == [b"first", b"second"]


@pytest.mark.asyncio
async def test_vad_stop_sends_finalize_after_audio(monkeypatch):
    service, _ = _service(monkeypatch)

    async for _ in service.run_stt(b"audio"):
        pass
    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert service._websocket.sent == [b"audio", FINALIZE_MESSAGE]


@pytest.mark.asyncio
async def test_each_vad_stop_sends_finalize(monkeypatch):
    service, _ = _service(monkeypatch)

    async for _ in service.run_stt(b"audio"):
        pass
    frame = VADUserStoppedSpeakingFrame()
    await service.process_frame(frame, FrameDirection.DOWNSTREAM)
    await service.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert service._websocket.sent == [b"audio", FINALIZE_MESSAGE, FINALIZE_MESSAGE]


@pytest.mark.asyncio
async def test_vad_finalize_can_be_disabled(monkeypatch):
    service, _ = _service(monkeypatch, vad_force_turn_endpoint=False)

    async for _ in service.run_stt(b"audio"):
        pass
    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert service._websocket.sent == [b"audio"]


@pytest.mark.asyncio
async def test_each_partial_immediately_emits_the_full_segment(monkeypatch):
    service, pushed = _service(
        monkeypatch,
        messages=[
            _transcription("Hello ", is_eos=False, segment="Hello "),
            _transcription("world", is_eos=False, segment="Hello world"),
        ],
    )

    await service._receive_messages()

    assert [frame.text for frame in pushed] == ["Hello ", "Hello world"]
    assert all(isinstance(frame, InterimTranscriptionFrame) for frame in pushed)
    assert pushed[-1].result["delta"]["text"] == "world"
    assert pushed[-1].result["transcription_id"] == "segment-1"


@pytest.mark.asyncio
async def test_delta_with_marker_is_finalized_and_stripped(monkeypatch):
    service, pushed = _service(
        monkeypatch,
        messages=[_transcription("world.<fin>", is_eos=True, segment="Hello world.<fin>")],
    )

    await service._send_finalize()
    await service._receive_messages()

    assert len(pushed) == 1
    assert isinstance(pushed[0], TranscriptionFrame)
    assert pushed[0].text == "Hello world."
    assert pushed[0].language == Language.EN
    assert pushed[0].finalized is True


@pytest.mark.asyncio
async def test_marker_on_segment_only_finalizes_the_segment(monkeypatch):
    service, pushed = _service(
        monkeypatch,
        messages=[_transcription("world.", is_eos=False, segment="Hello world.<fin>")],
    )

    await service._send_finalize()
    await service._receive_messages()

    assert isinstance(pushed[0], TranscriptionFrame)
    assert pushed[0].text == "Hello world."
    assert pushed[0].finalized is True


@pytest.mark.asyncio
async def test_automatic_end_of_segment_remains_interim(monkeypatch):
    service, pushed = _service(
        monkeypatch,
        messages=[_transcription("Hallo Welt.", is_eos=True, language="de")],
    )

    await service._receive_messages()

    assert len(pushed) == 1
    assert isinstance(pushed[0], InterimTranscriptionFrame)
    assert pushed[0].text == "Hallo Welt."
    assert pushed[0].language == Language.DE


@pytest.mark.asyncio
async def test_marker_only_response_does_not_emit_empty_frame(monkeypatch):
    service, pushed = _service(
        monkeypatch,
        messages=[_transcription("<fin>", is_eos=False)],
    )

    await service._receive_messages()

    assert pushed == []


@pytest.mark.asyncio
async def test_marker_only_response_does_not_republish_interim_text(monkeypatch):
    service, pushed = _service(
        monkeypatch,
        messages=[
            _transcription("Hello", is_eos=False, language="en"),
            _transcription("<fin>", is_eos=False, language=""),
            _transcription("<fin>", is_eos=False),
        ],
    )

    await service._send_finalize()
    await service._receive_messages()

    assert [type(frame) for frame in pushed] == [InterimTranscriptionFrame]
    assert [frame.text for frame in pushed] == ["Hello"]
    assert pushed[-1].language == Language.EN


@pytest.mark.asyncio
async def test_eos_without_new_delta_emits_full_interim_segment(monkeypatch):
    service, pushed = _service(
        monkeypatch,
        messages=[
            _transcription("Hello", is_eos=False),
            _transcription("", is_eos=True, segment="Hello"),
            _transcription("<fin>", is_eos=False),
        ],
    )

    await service._receive_messages()

    assert [type(frame) for frame in pushed] == [
        InterimTranscriptionFrame,
        InterimTranscriptionFrame,
    ]
    assert [frame.text for frame in pushed] == ["Hello", "Hello"]
    assert pushed[-1].result["delta"]["text"] == ""


@pytest.mark.asyncio
async def test_standard_rtvi_observer_receives_partial_then_final_text(monkeypatch):
    service, pushed = _service(
        monkeypatch,
        messages=[
            _transcription("Hel", is_eos=False),
            _transcription("lo", is_eos=False, segment="Hello"),
            _transcription("<fin>", is_eos=False, segment="Hello<fin>"),
        ],
    )
    rtvi = RTVIProcessor()
    send = AsyncMock()
    monkeypatch.setattr(rtvi, "push_transport_message", send)
    observer = RTVIObserver(rtvi)

    await service._receive_messages()
    for frame in pushed:
        await observer._handle_user_transcriptions(frame)

    messages = [call.args[0].data for call in send.call_args_list]
    assert [(message.text, message.final) for message in messages] == [
        ("Hel", False),
        ("Hello", False),
        ("Hello", True),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("marker_with_text", [False, True])
async def test_only_nonempty_final_segments_update_transcript_timing(monkeypatch, marker_with_text):
    service, _ = _service(monkeypatch)
    monkeypatch.setattr(service, "stop_ttfb_metrics", AsyncMock())
    monkeypatch.setattr(service, "_cancel_ttfb_timeout", AsyncMock())
    await service._send_finalize()

    messages = [
        _transcription("Hel", is_eos=False),
        _transcription("lo", is_eos=False, segment="Hello"),
        _transcription(" ", is_eos=False, segment="Hello "),
        _transcription(
            "world.<fin>" if marker_with_text else "<fin>",
            is_eos=marker_with_text,
            segment="Hello world.<fin>" if marker_with_text else "<fin>",
        ),
        _transcription("Next", is_eos=False),
        _transcription("", is_eos=True, segment="Next"),
        _transcription("<fin>", is_eos=False),
    ]
    for index, raw in enumerate(messages):
        monkeypatch.setattr(
            "pipecat.services.stt_service.time.time", lambda index=index: 100.0 + index
        )
        await service._handle_transcription_message(json.loads(raw))

        has_final_text = marker_with_text and index >= 3
        assert service._last_transcript_time == (103.0 if has_final_text else 0)
        assert service.stop_ttfb_metrics.await_count == int(has_final_text)
        assert service._cancel_ttfb_timeout.await_count == int(has_final_text)


@pytest.mark.asyncio
async def test_new_audio_after_finalize_can_be_finalized_again(monkeypatch):
    service, _ = _service(monkeypatch)

    async for _ in service.run_stt(b"first"):
        pass
    await service._send_finalize()
    async for _ in service.run_stt(b"second"):
        pass
    await service._send_finalize()

    assert service._websocket.sent == [
        b"first",
        FINALIZE_MESSAGE,
        b"second",
        FINALIZE_MESSAGE,
    ]


@pytest.mark.asyncio
async def test_unknown_and_malformed_messages_are_ignored(monkeypatch):
    service, pushed = _service(
        monkeypatch,
        messages=["not json", json.dumps({"message_type": "unknown"})],
    )

    await service._receive_messages()

    assert pushed == []


@pytest.mark.asyncio
async def test_translated_transcription_becomes_a_translation_frame(monkeypatch):
    service, pushed = _service(
        monkeypatch,
        messages=[
            json.dumps(
                {
                    "message_type": "translated_transcription",
                    "language": "es",
                    "segment": {"text": "Hola mundo"},
                }
            )
        ],
    )

    await service._receive_messages()

    assert len(pushed) == 1
    assert isinstance(pushed[0], TranslationFrame)
    assert pushed[0].text == "Hola mundo"
    assert pushed[0].language == Language.ES
