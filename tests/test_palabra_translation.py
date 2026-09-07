#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json

import pytest
from websockets.protocol import State

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TranslationFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.palabra.translation import (
    OUTPUT_SAMPLE_RATE,
    PalabraTranslationService,
)
from pipecat.transcriptions.language import Language


class _FakeWebsocket:
    def __init__(self, messages=(), *, state=State.OPEN):
        self._messages = list(messages)
        self.state = state
        self.sent = []

    async def send(self, payload):
        self.sent.append(json.loads(payload))

    async def close(self):
        self.state = State.CLOSED

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for message in self._messages:
            yield message


def _settings(**kwargs) -> PalabraTranslationService.Settings:
    kwargs.setdefault("target_languages", [Language.ES])
    return PalabraTranslationService.Settings(**kwargs)


def _service(monkeypatch=None, messages=(), ready=True, **kwargs):
    kwargs.setdefault("settings", _settings())
    service = PalabraTranslationService(api_key="test-key", **kwargs)
    service._sample_rate = 16000
    service._input_sample_rate = 16000
    service._websocket = _FakeWebsocket(messages)
    if ready:
        service._task_ready.set()

    pushed = []
    errors = []

    async def fake_push_frame(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    async def fake_push_error(error_msg, **kwargs):
        errors.append(error_msg)

    async def fake_noop(*args, **kwargs):
        pass

    if monkeypatch:
        monkeypatch.setattr(service, "push_frame", fake_push_frame)
        monkeypatch.setattr(service, "push_error", fake_push_error)
        monkeypatch.setattr(service, "_handle_transcription", fake_noop)
        monkeypatch.setattr(service, "emit_stt_usage_metrics", fake_noop)
    return service, pushed, errors


def _msg(message_type: str, data: dict) -> str:
    return json.dumps({"message_type": message_type, "data": data})


def _transcript(message_type: str, text: str, language: str, transcription_id="t1") -> str:
    return _msg(
        message_type,
        {
            "transcription": {
                "transcription_id": transcription_id,
                "language": language,
                "text": text,
                "segments": [],
            }
        },
    )


def _audio(audio: bytes, *, transcription_id="t1", language="es", last_chunk=False) -> str:
    return _msg(
        "output_audio_data",
        {
            "transcription_id": transcription_id,
            "language": language,
            "last_chunk": last_chunk,
            "data": base64.b64encode(audio).decode(),
        },
    )


def test_target_languages_are_required():
    with pytest.raises(ValueError):
        PalabraTranslationService(api_key="k")
    with pytest.raises(ValueError):
        PalabraTranslationService(
            api_key="k", settings=PalabraTranslationService.Settings(target_languages=[])
        )


def test_task_declares_streams_and_pipeline():
    service, *_ = _service(
        settings=_settings(
            language=Language.EN_US,
            target_languages=[Language.ES, Language.FR_CA],
            voice="voice-1",
            silence_threshold=0.7,
        )
    )

    task = service._build_task()

    assert task["input_stream"] == {
        "content_type": "audio",
        "source": {"type": "ws", "format": "pcm_s16le", "sample_rate": 16000, "channels": 1},
    }
    assert task["output_stream"] == {
        "content_type": "audio",
        "target": {"type": "ws", "format": "pcm_s16le"},
    }
    assert task["pipeline"] == {
        "transcription": {
            "source_language": "en-us",
            "segment_confirmation_silence_threshold": 0.7,
        },
        "translations": [
            {"target_language": "es", "speech_generation": {"voice_id": "voice-1"}},
            {"target_language": "fr-ca", "speech_generation": {"voice_id": "voice-1"}},
        ],
        "allowed_message_types": [
            "partial_transcription",
            "validated_transcription",
            "translated_transcription",
        ],
    }


def test_task_options_and_flags():
    service, *_ = _service(
        settings=_settings(
            language=None,
            generate_speech=False,
            voice="ignored",
            voice_cloning=True,
            translate_partials=True,
            transcription_options={"detectable_languages": ["en", "fr"]},
            translation_options={"allow_translation_glossaries": False},
            pipeline_options={"translation_queue_configs": {"global": {"auto_tempo": True}}},
        )
    )

    task = service._build_task()

    assert task["output_stream"] is None
    pipeline = task["pipeline"]
    assert pipeline["transcription"] == {
        "source_language": "auto",
        "detectable_languages": ["en", "fr"],
    }
    assert pipeline["translations"] == [
        {
            "target_language": "es",
            "speech_generation": {"voice_cloning": True},
            "translate_partial_transcriptions": True,
            "allow_translation_glossaries": False,
        }
    ]
    assert "partial_translated_transcription" in pipeline["allowed_message_types"]
    assert pipeline["translation_queue_configs"] == {"global": {"auto_tempo": True}}


@pytest.mark.asyncio
async def test_audio_is_dropped_until_the_task_is_ready():
    service, *_ = _service(ready=False, audio_chunk_ms=100)

    async for _ in service.run_stt(b"\x00" * 6400):
        pass

    assert service._websocket.sent == []
    assert service._audio_buffer == bytearray()


@pytest.mark.asyncio
async def test_audio_is_sent_as_base64_chunks():
    service, *_ = _service(audio_chunk_ms=100)
    chunk = 16000 * 2 * 100 // 1000

    async for _ in service.run_stt(b"\x01" * (chunk + 10)):
        pass

    sent = service._websocket.sent
    assert len(sent) == 1
    assert sent[0]["message_type"] == "input_audio_data"
    assert base64.b64decode(sent[0]["data"]["data"]) == b"\x01" * chunk
    assert len(service._audio_buffer) == 10


@pytest.mark.asyncio
async def test_current_task_marks_ready_and_fires_event(monkeypatch):
    service, pushed, errors = _service(
        monkeypatch,
        ready=False,
        messages=[
            _msg("error", {"code": "NOT_FOUND", "desc": "no task yet"}),
            _msg("current_task", {"task_status": "running"}),
            _msg("error", {"code": "VALIDATION_ERROR", "desc": "bad field"}),
        ],
    )
    events = []

    async def fake_call_event_handler(name, *args):
        events.append(name)

    monkeypatch.setattr(service, "_call_event_handler", fake_call_event_handler)

    await service._receive_messages()

    assert service.task_ready
    assert events == ["on_task_ready"]
    # NOT_FOUND while polling is expected; later errors are reported.
    assert errors == ["Palabra translation error VALIDATION_ERROR: bad field"]


@pytest.mark.asyncio
async def test_transcriptions_and_translations_become_frames(monkeypatch):
    service, pushed, _ = _service(
        monkeypatch,
        messages=[
            _transcript("partial_transcription", "The weather", "en"),
            _transcript("validated_transcription", "The weather is nice.", "en"),
            _transcript("partial_translated_transcription", "Hace", "es"),
            _transcript("translated_transcription", "Hace buen tiempo.", "es"),
        ],
    )

    await service._receive_messages()

    assert [type(f) for f in pushed] == [
        InterimTranscriptionFrame,
        TranscriptionFrame,
        TranslationFrame,
    ]
    assert pushed[0].text == "The weather"
    assert pushed[1].text == "The weather is nice."
    assert pushed[1].language == Language.EN
    assert pushed[2].text == "Hace buen tiempo."
    assert pushed[2].language == Language.ES


@pytest.mark.asyncio
async def test_output_audio_is_wrapped_in_tts_start_and_stop(monkeypatch):
    service, pushed, _ = _service(
        monkeypatch,
        messages=[
            _audio(b"\x01\x02"),
            _audio(b"\x03\x04", last_chunk=True),
            _audio(b"\x05\x06", transcription_id="t2"),
            _audio(b"", transcription_id="t2", last_chunk=True),
        ],
    )

    await service._receive_messages()

    assert [type(f) for f in pushed] == [
        TTSStartedFrame,
        TTSAudioRawFrame,
        TTSAudioRawFrame,
        TTSStoppedFrame,
        TTSStartedFrame,
        TTSAudioRawFrame,
        TTSStoppedFrame,
    ]
    assert pushed[1].audio == b"\x01\x02"
    assert pushed[1].sample_rate == OUTPUT_SAMPLE_RATE
    assert pushed[1].num_channels == 1
    assert pushed[0].context_id == pushed[1].context_id == pushed[3].context_id
    assert pushed[4].context_id != pushed[0].context_id


@pytest.mark.asyncio
async def test_overlapping_languages_share_one_speech_stream(monkeypatch):
    service, pushed, _ = _service(
        monkeypatch,
        messages=[
            _audio(b"\x01", language="es"),
            _audio(b"\x02", language="fr"),
            _audio(b"", language="es", last_chunk=True),
            _audio(b"", language="fr", last_chunk=True),
        ],
    )

    await service._receive_messages()

    assert [type(f) for f in pushed] == [
        TTSStartedFrame,
        TTSAudioRawFrame,
        TTSAudioRawFrame,
        TTSStoppedFrame,
    ]


@pytest.mark.asyncio
async def test_audio_destinations_route_each_language_to_its_own_track(monkeypatch):
    service, pushed, _ = _service(
        monkeypatch,
        settings=_settings(
            target_languages=[Language.ES, Language.FR, Language.DE],
            audio_destinations={Language.ES: None, Language.FR: "fr"},
        ),
        messages=[
            _audio(b"\x01", language="es"),
            _audio(b"\x02", language="fr"),
            _audio(b"\x03", language="de"),
            _audio(b"", language="es", last_chunk=True),
            _audio(b"", language="de", last_chunk=True),
            _audio(b"", language="fr", last_chunk=True),
        ],
    )

    await service._receive_messages()

    assert [(type(f).__name__, f.transport_destination) for f in pushed] == [
        ("TTSStartedFrame", None),
        ("TTSAudioRawFrame", None),
        ("TTSStartedFrame", "fr"),
        ("TTSAudioRawFrame", "fr"),
        ("TTSStoppedFrame", None),
        ("TTSStoppedFrame", "fr"),
    ]
    # German is not in the mapping, so its audio is dropped.
    assert all(f.audio != b"\x03" for f in pushed if isinstance(f, TTSAudioRawFrame))
    # Each destination has its own speech context.
    assert pushed[0].context_id == pushed[1].context_id == pushed[4].context_id
    assert pushed[2].context_id == pushed[3].context_id == pushed[5].context_id
    assert pushed[0].context_id != pushed[2].context_id


@pytest.mark.asyncio
async def test_end_of_stream_closes_every_destination(monkeypatch):
    service, pushed, _ = _service(
        monkeypatch,
        settings=_settings(
            target_languages=[Language.ES, Language.FR],
            audio_destinations={Language.ES: "es", Language.FR: "fr"},
        ),
        messages=[
            _audio(b"\x01", language="es"),
            _audio(b"\x02", language="fr"),
            _msg("end_of_stream", {}),
        ],
    )

    await service._receive_messages()

    stopped = [f.transport_destination for f in pushed if isinstance(f, TTSStoppedFrame)]
    assert sorted(stopped) == ["es", "fr"]


@pytest.mark.asyncio
async def test_end_of_stream_closes_speech_and_sets_event(monkeypatch):
    service, pushed, _ = _service(
        monkeypatch,
        messages=[_audio(b"\x01"), _msg("end_of_stream", {})],
    )

    await service._receive_messages()

    assert service._end_of_stream.is_set()
    assert isinstance(pushed[-1], TTSStoppedFrame)


@pytest.mark.asyncio
async def test_settings_change_resends_the_task(monkeypatch):
    service, *_ = _service(monkeypatch)
    service._is_usable = True

    await service._update_settings(_settings(target_languages=[Language.DE], voice="v2"))

    sent = service._websocket.sent
    assert len(sent) == 1
    assert sent[0]["message_type"] == "set_task"
    assert sent[0]["data"]["pipeline"]["translations"] == [
        {"target_language": "de", "speech_generation": {"voice_id": "v2"}}
    ]


@pytest.mark.asyncio
async def test_interrupt_and_speak_send_commands():
    service, *_ = _service()

    await service.interrupt()
    await service.interrupt([Language.ES], pause=True)
    await service.speak("Hola", Language.ES)
    await service.speak("Hello", Language.EN, translate=True)

    assert service._websocket.sent == [
        {"message_type": "interrupt_task", "data": {"languages": ["global"], "pause_task": False}},
        {"message_type": "interrupt_task", "data": {"languages": ["es"], "pause_task": True}},
        {
            "message_type": "tts_task",
            "data": {"text": "Hola", "language": "es", "translate_text": False},
        },
        {
            "message_type": "tts_task",
            "data": {"text": "Hello", "language": "en", "translate_text": True},
        },
    ]
