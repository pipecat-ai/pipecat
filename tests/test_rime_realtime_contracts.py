#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Output timing and shared frame behavior used by Prism conversations."""

import asyncio
import struct

import pytest

pytest.importorskip("rimelabs_sdk")

from pipecat.frames.frames import (
    AudioOutputEndFrame,
    AudioOutputGainFrame,
    AudioOutputStartFrame,
    AudioOutputStopFrame,
    AudioPlaybackReportFrame,
    InterruptionFrame,
    LLMInputTextFrame,
    LLMRunFrame,
    LLMUpdateSettingsFrame,
    OutputAudioRawFrame,
    UserStartedSpeakingFrame,
    VolumeFrame,
    VolumeGainFrame,
)
from pipecat.services.rime.realtime.llm import RimeRealtimeLLMService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.local._playback_clock import PlaybackClock

from .rime_realtime_peer import MeasuredOutput, conversation, until


def audio(context_id, samples=960, *, interruptible=True):
    frame = OutputAudioRawFrame(struct.pack("<h", 10000) * samples, 24000, 1, context_id=context_id)
    frame.interruptible = interruptible
    return frame


@pytest.mark.asyncio
async def test_hint_expiry_preserves_application_volume_and_gain():
    output = MeasuredOutput()
    await run_test(
        output,
        frames_to_send=[
            VolumeFrame(0.8),
            VolumeGainFrame(0.5),
            AudioOutputGainFrame(0.25, 0.05),
            AudioOutputStartFrame("quiet"),
            audio("quiet"),
            AudioOutputEndFrame("quiet"),
            SleepFrame(0.02),
            VolumeGainFrame(0.25),
            AudioOutputGainFrame(0.25, 0.05),
            SleepFrame(0.07),
            AudioOutputStartFrame("restored"),
            audio("restored"),
            AudioOutputEndFrame("restored"),
            SleepFrame(0.02),
        ],
    )
    assert struct.unpack_from("<h", output.writes[0].audio)[0] == 1000
    assert struct.unpack_from("<h", output.writes[1].audio)[0] == 2000


@pytest.mark.asyncio
async def test_scoped_stop_keeps_uninterruptible_and_unrelated_output():
    output = MeasuredOutput()
    down, up = await run_test(
        output,
        frames_to_send=[
            AudioOutputStartFrame("retained"),
            audio("retained", interruptible=False),
            AudioOutputStopFrame("retained", "user"),
            AudioOutputEndFrame("retained"),
            AudioOutputStartFrame("later"),
            audio("later"),
            AudioOutputEndFrame("later"),
            SleepFrame(0.03),
        ],
    )
    reports = [f for f in up if isinstance(f, AudioPlaybackReportFrame)]
    assert {f.context_id for f in reports} == {"retained", "later"}
    assert all(f.played_ms == 40 for f in reports)
    assert {f.context_id for f in output.writes if f.context_id} == {"retained", "later"}


def test_portaudio_clock_counts_only_elapsed_speech_spans():
    clock = PlaybackClock()
    clock.schedule("a", 1.0, 0.01, valid=True)
    clock.schedule("a", 1.1, 0.02, valid=True)
    assert clock.played_ms("a", 0.9) == 0
    assert clock.played_ms("a", 1.105) == pytest.approx(15)
    assert clock.played_ms("a", 1.5) == pytest.approx(30)
    clock.schedule("b", 2.0, 0.02, valid=False)
    assert clock.played_ms("b", 2.1) is None
    clock.forget("a")
    assert clock.played_ms("a", 3.0) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("opening,later", [(True, False), (False, True)])
async def test_reply_keeps_starting_interruption_policy(opening, later):
    async with conversation(
        settings=RimeRealtimeLLMService.Settings(interrupt_on_speech=opening)
    ) as (peer, service, worker, ctx, output, up, down):
        await worker.queue_frame(LLMRunFrame())
        ref = await peer.start(await peer.next())
        await peer.message(ref, samples=1920, done=False)
        await until(lambda: bool(output.writes))
        update = LLMUpdateSettingsFrame(delta=service.Settings(interrupt_on_speech=later))
        await worker.queue_frame(update)
        from .test_rime_realtime_service import results

        await until(lambda: results(up, update))
        await peer.emit("input_audio_buffer.speech_started", item_id="speech", audio_start_ms=0)
        await until(lambda: any(isinstance(f, UserStartedSpeakingFrame) for f in up))
        await asyncio.sleep(0.02)
        cancellations = [e for e in peer.events if e["type"] == "response.cancel"]
        assert bool(cancellations) == opening
        interruptions = [f for f in up if isinstance(f, InterruptionFrame)]
        assert len(interruptions) == int(opening)
        if not opening:
            await peer.end(ref)
        await asyncio.wait_for(peer.receipts.get(), 3)


@pytest.mark.asyncio
async def test_typed_audio_option_is_bound_to_its_accepted_response():
    async with conversation() as (peer, service, worker, ctx, output, up, down):
        await worker.queue_frame(LLMInputTextFrame("Text reply", audio_response=False))
        ref = await peer.start(await peer.next())
        await peer.message(ref)
        await peer.end(ref)
        assert (await asyncio.wait_for(peer.receipts.get(), 3))["played_ms"] == 0
        assert not output.writes
        await worker.queue_frame(LLMInputTextFrame("Spoken reply", audio_response=True))
        ref = await peer.start(await peer.next())
        await peer.message(ref)
        await peer.end(ref)
        assert (await asyncio.wait_for(peer.receipts.get(), 3))["played_ms"] > 0
        assert output.writes


@pytest.mark.asyncio
async def test_delayed_playback_report_reaches_the_inactive_owner():
    from pipecat.frames.frames import Frame, ManuallySwitchServiceFrame
    from pipecat.pipeline.service_switcher import ServiceSwitcher
    from pipecat.processors.frame_processor import FrameProcessor

    class Owner(FrameProcessor):
        def __init__(self, context_id):
            super().__init__()
            self.context_id = context_id
            self.reports = []

        def owns_audio_output(self, context_id):
            return context_id == self.context_id

        async def process_frame(self, frame: Frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, AudioPlaybackReportFrame):
                self.reports.append(frame)
            else:
                await self.push_frame(frame, direction)

    old, new = Owner("old-output"), Owner("new-output")
    switcher = ServiceSwitcher([old, new])
    from pipecat.processors.frame_processor import FrameDirection

    await run_test(
        switcher,
        frames_to_send=[
            ManuallySwitchServiceFrame(service=new),
            AudioPlaybackReportFrame("old-output", "completed", 20),
            AudioPlaybackReportFrame("new-output", "completed", 30),
            SleepFrame(0.02),
        ],
        frames_to_send_direction=FrameDirection.UPSTREAM,
    )
    assert [report.context_id for report in old.reports] == ["old-output"]
    assert [report.context_id for report in new.reports] == ["new-output"]


@pytest.mark.asyncio
async def test_local_callback_path_reports_dac_samples_without_padding():
    pytest.importorskip("pyaudio")
    from pipecat.transports.local.audio import LocalAudioOutputTransport, LocalAudioTransportParams

    class Device:
        def __init__(self):
            self.now = 10.0
            self.callback = None
            self.closed = False
            self.buffers = []

        def get_format_from_width(self, width):
            return width

        def open(self, **options):
            self.callback = options["stream_callback"]
            return self

        def start_stream(self):
            pass

        def stop_stream(self):
            pass

        def close(self):
            self.closed = True

        def get_time(self):
            return self.now

        async def run(self):
            await until(lambda: self.callback is not None)
            while not self.closed:
                self.now += 0.04
                pcm, _ = self.callback(None, 960, {"output_buffer_dac_time": self.now + 0.01}, 0)
                self.buffers.append(pcm)
                await asyncio.sleep(0.001)

    device = Device()
    output = LocalAudioOutputTransport(
        device,
        LocalAudioTransportParams(
            audio_out_enabled=True, audio_out_sample_rate=24000, audio_out_playback_clock=True
        ),
    )
    result, _ = await asyncio.gather(
        run_test(
            output,
            frames_to_send=[
                AudioOutputStartFrame("device"),
                audio("device", samples=241),
                AudioOutputEndFrame("device"),
                SleepFrame(0.03),
            ],
        ),
        device.run(),
    )
    down, up = result
    reports = [f for f in up if isinstance(f, AudioPlaybackReportFrame)]
    assert len(reports) == 1
    assert reports[0].played_ms == pytest.approx(241 / 24)
    assert reports[0].outcome == "completed"
    assert any(struct.pack("<h", 10000) * 241 in buffer for buffer in device.buffers)


@pytest.mark.asyncio
async def test_explicit_typed_audio_option_survives_output_toggle_restoration():
    from pipecat.frames.frames import LLMConfigureOutputFrame

    async with conversation() as (peer, service, worker, ctx, output, up, down):
        await worker.queue_frame(LLMConfigureOutputFrame(skip_tts=False))
        await worker.queue_frame(LLMInputTextFrame("Speak this reply", audio_response=True))
        await worker.queue_frame(LLMConfigureOutputFrame(skip_tts=True))
        ref = await peer.start(await peer.next())
        await peer.message(ref)
        await peer.end(ref)
        assert (await asyncio.wait_for(peer.receipts.get(), 3))["played_ms"] > 0


@pytest.mark.asyncio
async def test_incompatible_tracked_output_channels_fail_explicitly():
    from pipecat.frames.frames import ErrorFrame

    output = MeasuredOutput(channels=2)
    down, up = await run_test(
        output,
        frames_to_send=[
            AudioOutputStartFrame("mono"),
            audio("mono"),
            AudioOutputEndFrame("mono"),
            SleepFrame(0.02),
        ],
    )
    reports = [frame for frame in up if isinstance(frame, AudioPlaybackReportFrame)]
    assert len(reports) == 1
    assert reports[0].outcome == "failed"
    assert reports[0].played_ms == 0
    assert any(isinstance(frame, ErrorFrame) for frame in up)
    assert not [frame for frame in output.writes if frame.context_id]
