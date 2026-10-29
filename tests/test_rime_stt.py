#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Rime settings, optional dependency and completed-segment recognition."""

import asyncio
import io
import subprocess
import sys
import time
import wave
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from rimelabs_sdk import RimeAuthenticationError, RimeInputError, Transcription

from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    InterruptionFrame,
    MetricsFrame,
    STTMuteFrame,
    STTUpdateSettingsFrame,
    UserAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import ProcessingMetricsData
from pipecat.services.rime import stt as rime
from pipecat.services.rime.stt import RimeSegmentedSTTService, RimeSTTService
from pipecat.tests.utils import run_test
from pipecat.transcriptions.language import Language
from pipecat.utils.types import is_given
from tests.rime_stt_helpers import FakeRime, audio, harness, send


@pytest.mark.parametrize("service_cls", [RimeSTTService, RimeSegmentedSTTService])
def test_settings_are_complete_copied_and_construction_is_lazy(monkeypatch, service_cls):
    constructor = Mock(side_effect=AssertionError("construction must not create a client"))
    monkeypatch.setattr(rime, "Rime", constructor)
    terms = [" Pipecat ", "Rime", "Rime"]
    settings = service_cls.Settings(language=Language.EN_US, context_terms=terms)
    service = service_cls(api_key="test", settings=settings)
    terms.append("later")
    assert service._settings.context_terms == (" Pipecat ", "Rime", "Rime")
    assert service._settings.language == "en-US"
    assert service._settings.model is None
    assert not constructor.called
    assert all(is_given(getattr(service._settings, f.name)) for f in fields(service._settings))
    assert all(
        not is_given(getattr(service_cls.Settings(), f.name))
        for f in fields(settings)
        if f.name != "extra"
    )


@pytest.mark.parametrize("service_cls", [RimeSTTService, RimeSegmentedSTTService])
@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
@pytest.mark.parametrize("parameter", ["request_timeout", "shutdown_timeout"])
def test_invalid_timeout(service_cls, timeout, parameter):
    with pytest.raises(ValueError, match=parameter):
        service_cls(api_key="test", **{parameter: timeout})


@pytest.mark.parametrize("service_cls", [RimeSTTService, RimeSegmentedSTTService])
def test_client_and_api_key_are_mutually_exclusive(service_cls):
    with pytest.raises(ValueError, match="either client or api_key"):
        service_cls(client=FakeRime(), api_key="test")


@pytest.mark.asyncio
@pytest.mark.parametrize("service_cls", [RimeSTTService, RimeSegmentedSTTService])
async def test_injected_client_survives_shutdown_and_repeated_cleanup(service_cls):
    client = FakeRime(automatic=True)
    service = service_cls(client=client)
    await run_test(
        service,
        frames_to_send=[VADUserStartedSpeakingFrame(), audio(), VADUserStoppedSpeakingFrame()],
    )
    await service.cleanup()
    assert not client.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("service_cls", [RimeSTTService, RimeSegmentedSTTService])
async def test_settings_update_is_atomic(service_cls):
    service = service_cls(api_key="test")
    with pytest.raises(ValueError, match="mode"):
        await service._update_settings(service.Settings(language="es-MX", mode="invalid"))
    assert service._settings.language == "en"
    await service._update_settings(service.Settings(language="es-MX"))
    assert service._settings.language == "es-MX"
    with pytest.raises(ValueError, match="model"):
        await service._update_settings(service.Settings(model="unknown"))
    assert service._settings.model is None


@pytest.mark.asyncio
@pytest.mark.parametrize("service_cls", [RimeSTTService, RimeSegmentedSTTService])
async def test_partial_startup_reports_error_and_cleans_up(monkeypatch, service_cls):
    monkeypatch.setattr(
        rime, "Rime", Mock(side_effect=RimeAuthenticationError("missing credential"))
    )
    service = service_cls(api_key="")
    _, upstream = await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            audio(),
            VADUserStoppedSpeakingFrame(),
        ],
    )
    errors = [frame for frame in upstream if isinstance(frame, ErrorFrame)]
    assert len(errors) == 1
    assert not service.is_usable
    await service.cleanup()


def test_missing_sdk_installation_hint():
    code = """
import sys
sys.modules['rimelabs_sdk'] = None
import pipecat.services.rime
try:
    import pipecat.services.rime.stt
except ImportError as error:
    assert 'pipecat-ai[rime-stt]' in str(error)
else:
    raise AssertionError('missing SDK must fail with an installation hint')
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [8000, 16000, 24000, 48000])
@pytest.mark.parametrize("channels", [1, 2])
async def test_segment_normalizes_wav_samples(monkeypatch, rate, channels):
    client = FakeRime(automatic=True)
    monkeypatch.setattr(rime, "Rime", lambda **kwargs: client)
    service = RimeSegmentedSTTService(api_key="test", trailing_silence_secs=0)
    samples = np.full((rate // 10, channels), 1000, dtype="<i2")
    if channels == 2:
        samples[:, 1] = 3000
    pcm = samples.tobytes()
    await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            audio(pcm[:3], rate, channels),
            audio(pcm[3:], rate, channels),
            VADUserStoppedSpeakingFrame(),
        ],
    )
    wav_bytes, options, _ = client.calls[0]
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        actual = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    assert len(actual) == 1600
    assert actual[100:-100].mean() == pytest.approx(2000 if channels == 2 else 1000, abs=1)
    assert options["language"] == "en"
    assert "output_contract" not in options
    assert client.closed


@pytest.mark.asyncio
async def test_segment_snapshots_settings_and_speaker(monkeypatch):
    client = FakeRime()
    monkeypatch.setattr(rime, "Rime", lambda **kwargs: client)
    service = RimeSegmentedSTTService(api_key="test", trailing_silence_secs=0)
    async with harness(service) as (out, _):
        await send(
            service,
            VADUserStartedSpeakingFrame(),
            UserAudioRawFrame(
                audio=b"\0\0" * 160, sample_rate=16000, num_channels=1, user_id="first"
            ),
            VADUserStoppedSpeakingFrame(),
        )
        async with asyncio.timeout(2):
            call_a = await client.new_call.get()
        await send(
            service,
            VADUserStartedSpeakingFrame(),
            audio(),
            STTUpdateSettingsFrame(delta=service.Settings(language="es")),
            VADUserStoppedSpeakingFrame(),
        )
        call_a[2].set_result(Transcription(text="first", language="en", request_id="a"))
        async with asyncio.timeout(2):
            call_b = await client.new_call.get()
        assert call_b[1]["language"] == "en"
        call_b[2].set_result(Transcription(text="second", language="en", request_id="b"))
        frames = await out.wait_for(rime.TranscriptionFrame, 2)
        assert [f.user_id for f in frames] == ["first", ""]
        assert [f.finalized for f in frames] == [False, True]
        assert [f.result["request_id"] for f in frames] == ["a", "b"]


@pytest.mark.asyncio
async def test_segment_failure_during_capture_allows_settings_recovery():
    client = FakeRime()
    service = RimeSegmentedSTTService(client=client, trailing_silence_secs=0)
    async with harness(service) as (out, upstream):
        await send(service, VADUserStartedSpeakingFrame(), audio(), VADUserStoppedSpeakingFrame())
        async with asyncio.timeout(2):
            failed = await client.new_call.get()

        await send(service, VADUserStartedSpeakingFrame(), audio())
        failed[2].set_exception(RimeInputError("Invalid recognition setting"))
        await upstream.wait_for(ErrorFrame)
        assert not service.is_usable
        await send(service, VADUserStoppedSpeakingFrame())

        await send(service, STTUpdateSettingsFrame(delta=service.Settings(language="es")))
        assert service.is_usable
        await send(service, VADUserStartedSpeakingFrame(), audio(), VADUserStoppedSpeakingFrame())
        async with asyncio.timeout(2):
            recovered = await client.new_call.get()
        assert recovered[1]["language"] == "es"
        recovered[2].set_result(Transcription(text="recovered", language="es"))

        frames = await out.wait_for(rime.TranscriptionFrame)
        assert [frame.text for frame in frames] == ["recovered"]
        assert frames[0].finalized
        assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_segment_mute_discards_queued_speech_and_recovers_without_vad_start(monkeypatch):
    client = FakeRime()
    monkeypatch.setattr(rime, "Rime", lambda **kwargs: client)
    service = RimeSegmentedSTTService(api_key="test", trailing_silence_secs=0)
    async with harness(service) as (out, _):
        await send(service, VADUserStartedSpeakingFrame(), audio(), VADUserStoppedSpeakingFrame())
        async with asyncio.timeout(2):
            old = await client.new_call.get()
        await send(
            service,
            STTMuteFrame(mute=True),
            VADUserStartedSpeakingFrame(),
            audio(),
            STTMuteFrame(mute=False),
            audio(),
            VADUserStoppedSpeakingFrame(),
        )
        async with asyncio.timeout(2):
            new = await client.new_call.get()
        assert old[2].cancelled()
        new[2].set_result(Transcription(text="fresh", language="en"))
        frames = await out.wait_for(rime.TranscriptionFrame)
        assert [f.text for f in frames] == ["fresh"]


@pytest.mark.asyncio
async def test_segment_limit_and_graceful_active_speech(monkeypatch):
    client = FakeRime(automatic=True)
    monkeypatch.setattr(rime, "Rime", lambda **kwargs: client)
    monkeypatch.setattr(rime, "_MAX_AUDIO_SECONDS", 1)
    service = RimeSegmentedSTTService(api_key="test", trailing_silence_secs=0)
    async with harness(service) as (out, upstream):
        await send(
            service,
            VADUserStartedSpeakingFrame(),
            audio(bytes(32002)),
            audio(),
            VADUserStoppedSpeakingFrame(),
        )
        assert len(await upstream.wait_for(ErrorFrame)) == 1
        assert client.calls == []
        await send(service, VADUserStartedSpeakingFrame(), audio(), EndFrame())
        assert [f.text for f in out.transcripts] == ["segment 1"]
        assert client.closed


@pytest.mark.asyncio
async def test_segment_processing_timer_survives_interruption(monkeypatch):
    client = FakeRime()
    monkeypatch.setattr(rime, "Rime", lambda **kwargs: client)
    clock = [100.0]
    monkeypatch.setattr(rime, "time", SimpleNamespace(time=time.time, monotonic=lambda: clock[0]))
    service = RimeSegmentedSTTService(api_key="test")
    async with harness(service) as (out, _):
        await send(service, VADUserStartedSpeakingFrame(), audio(), VADUserStoppedSpeakingFrame())
        async with asyncio.timeout(2):
            call = await client.new_call.get()
        clock[0] = 102.0
        await send(service, InterruptionFrame())
        clock[0] = 105.0
        call[2].set_result(Transcription(text="complete", language="en"))
        await out.wait_for(rime.TranscriptionFrame)
        processing = [
            d
            for f in out.frames
            if isinstance(f, MetricsFrame)
            for d in f.data
            if isinstance(d, ProcessingMetricsData)
        ]
        assert [metric.value for metric in processing] == [5.0]
