#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame-level reproduction of AudioBufferProcessor startup-silence.

This script drives a real AudioBufferProcessor with Pipecat frames (no Daily,
no network, no API keys) to reproduce a customer report: the first 2-3 seconds
of a stereo recording sound choppy on the user (left) channel because a run of
inserted zero-silence appears near the start of the user track.

Two mechanisms in src/pipecat/processors/audio/audio_buffer_processor.py can
insert that leading zero-silence (both new in Pipecat 1.3.0, PR #4567):

  1. _fill_buffer_silence_gap(...): when the user buffer was last written more
     than 200 ms ago, it appends silence proportional to the wall-clock gap.
  2. _sync_buffer_to_position(...): when bot audio arrives while the user is
     silent, it zero-pads the user buffer so the two tracks stay aligned.

Customer config (matched here):
  - StartFrame(audio_in_sample_rate=16000, audio_out_sample_rate=24000)
  - AudioBufferProcessor(sample_rate=24000, num_channels=2)
  - user InputAudioRawFrame at 16000 Hz (resampled up to 24000 Hz)
  - bot OutputAudioRawFrame at 24000 Hz

The "real" audio segments are non-silent sine tones so any inserted exact-zero
silence is unambiguous to detect. We capture the recorded tracks via the
on_track_audio_data event handler and scan the user (left) channel for a
contiguous run of exact 0x0000 int16 samples longer than 200 ms within the
first ~1.5 s.

Run from the repo root:

    uv run python examples/audiobuffer-startup-silence-repro/repro.py

Determinism: case (a) monkeypatches time.monotonic in the processor module so
the wall-clock gap is controlled exactly (no real sleeps). The symbol patched
is pipecat.processors.audio.audio_buffer_processor.time, which is what the
module uses.

We keep this repro standalone (frame-level) rather than running it through the
pipecat.evals harness. The evals framework is built around end-to-end
scenarios with transports and TTS-generated speech, which is heavier than we
need to exercise the buffer logic in isolation. We still borrow the evals/test
idioms: setup() + StartFrame + start_recording() to drive a single processor,
and the time-module monkeypatch used by tests/test_audio_buffer_processor.py.
"""

import asyncio
import math
import struct
from types import SimpleNamespace
from unittest.mock import patch

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

# Customer config.
USER_IN_RATE = 16000  # user InputAudioRawFrame sample rate (resampled up)
BOT_OUT_RATE = 24000  # bot OutputAudioRawFrame sample rate
RECORD_RATE = 24000  # AudioBufferProcessor output sample rate
NUM_CHANNELS = 2  # stereo: user left, bot right

# A 20 ms frame is a typical WebRTC chunk.
FRAME_MS = 20

# Detection thresholds.
LEADING_WINDOW_S = 1.5  # only look at the first 1.5 s of the user track
MIN_SILENCE_S = 0.2  # a run longer than 200 ms is "inserted", not jitter


def make_tone(sample_rate: int, duration_ms: int, freq: int = 440, amplitude: int = 12000) -> bytes:
    """Generate a mono 16-bit PCM sine tone (non-silent, never exact zero except at phase zero).

    We use a small amplitude offset so samples are reliably non-zero, making
    inserted exact-zero silence trivial to distinguish from real audio.
    """
    n = int(sample_rate * duration_ms / 1000)
    out = bytearray()
    for i in range(n):
        # Sine plus a tiny DC-ish bias keeps values away from exact zero.
        sample = int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
        if sample == 0:
            sample = 1  # never write an exact-zero "real" sample
        out += struct.pack("<h", sample)
    return bytes(out)


async def make_processor() -> AudioBufferProcessor:
    """Create and start a processor with the customer's config, ready to record.

    Mirrors tests/test_audio_buffer_processor.py: setup() then a StartFrame
    through the public process_frame path, then start_recording(). The real
    resampler is kept (not stubbed) so the 16 kHz -> 24 kHz input path runs.
    """
    processor = AudioBufferProcessor(sample_rate=RECORD_RATE, num_channels=NUM_CHANNELS)

    loop = asyncio.get_event_loop()
    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=loop))
    await processor.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=task_manager,
            pipeline_worker=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
        )
    )

    await processor.process_frame(
        StartFrame(audio_in_sample_rate=USER_IN_RATE, audio_out_sample_rate=BOT_OUT_RATE),
        FrameDirection.DOWNSTREAM,
    )
    await processor.start_recording()
    return processor


async def send_user(processor: AudioBufferProcessor, audio: bytes):
    await processor.process_frame(
        InputAudioRawFrame(audio=audio, sample_rate=USER_IN_RATE, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )


async def send_bot(processor: AudioBufferProcessor, audio: bytes):
    await processor.process_frame(
        OutputAudioRawFrame(audio=audio, sample_rate=BOT_OUT_RATE, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )


async def capture_tracks(processor: AudioBufferProcessor) -> tuple[bytes, bytes]:
    """Flush the processor and return (user_track, bot_track) from on_track_audio_data."""
    captured = {}
    event = asyncio.Event()

    async def on_track_audio_data(_, user, bot, sample_rate, num_channels):
        captured["user"] = user
        captured["bot"] = bot
        event.set()

    processor.add_event_handler("on_track_audio_data", on_track_audio_data)
    await processor.stop_recording()
    await asyncio.wait_for(event.wait(), timeout=2)
    return captured["user"], captured["bot"]


def longest_leading_zero_run(track: bytes, sample_rate: int) -> tuple[int, float, float]:
    """Find the longest run of exact-zero int16 samples within the leading window.

    Returns (start_sample_index, run_seconds, start_seconds). A track is mono
    16-bit, so each sample is 2 bytes.
    """
    total_samples = len(track) // 2
    window_samples = min(total_samples, int(LEADING_WINDOW_S * sample_rate))

    samples = struct.unpack("<%dh" % total_samples, track[: total_samples * 2])

    best_len = 0
    best_start = 0
    cur_len = 0
    cur_start = 0
    for i in range(window_samples):
        if samples[i] == 0:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0

    return best_start, best_len / sample_rate, best_start / sample_rate


def report_case(name: str, user_track: bytes, sample_rate: int) -> bool:
    """Print findings for one case. Returns True if leading inserted silence found."""
    start_idx, run_s, start_s = longest_leading_zero_run(user_track, sample_rate)
    found = run_s > MIN_SILENCE_S

    print(
        f"  user track length:        {len(user_track) // 2} samples "
        f"({len(user_track) / 2 / sample_rate:.3f} s)"
    )
    print(
        f"  longest leading zero run: {run_s * 1000:.0f} ms "
        f"(starts at {start_s:.3f} s, sample index {start_idx})"
    )
    print(
        f"  threshold:                {MIN_SILENCE_S * 1000:.0f} ms within first "
        f"{LEADING_WINDOW_S:.1f} s"
    )
    print(f"  result:                   {'INSERTED SILENCE DETECTED' if found else 'clean'}")
    return found


async def case_control() -> bool:
    """CONTROL: steady, continuous user audio from t=0, interleaved with bot audio.

    The user buffer is written every frame with no wall-clock gap, so neither
    the gap-fill nor the bot-sync path should inject a long leading zero run.
    """
    print("CONTROL: continuous user audio from t=0 (expect: clean)")
    user_frame = make_tone(USER_IN_RATE, FRAME_MS, freq=440)
    bot_frame = make_tone(BOT_OUT_RATE, FRAME_MS, freq=880)

    fake = SimpleNamespace(monotonic=lambda: case_control.clock)
    case_control.clock = 0.0
    with patch("pipecat.processors.audio.audio_buffer_processor.time", fake):
        p = await make_processor()
        # 1.5 s of steady interleaved audio, advancing the fake clock 20 ms/frame.
        for _ in range(int(1.5 * 1000 / FRAME_MS)):
            await send_user(p, user_frame)
            await send_bot(p, bot_frame)
            case_control.clock += FRAME_MS / 1000
        user_track, _ = await capture_tracks(p)
        await p.cleanup()

    found = report_case("control", user_track, RECORD_RATE)
    print()
    return found


async def case_a_gap_fill() -> bool:
    """BUG (a): short user burst, then a >200 ms gap, then resume.

    This is the _fill_buffer_silence_gap path. The burst writes the user buffer
    and pins _last_user_buffer_update_time. We then advance the fake clock past
    the 200 ms threshold before the next user frame, so silence proportional to
    the gap is appended just after the burst.
    """
    print("BUG (a): user burst, >200 ms gap, resume (expect: inserted silence via gap-fill)")
    burst = make_tone(USER_IN_RATE, FRAME_MS, freq=440)
    resume = make_tone(USER_IN_RATE, FRAME_MS, freq=440)

    fake = SimpleNamespace(monotonic=lambda: case_a_gap_fill.clock)
    case_a_gap_fill.clock = 0.0
    with patch("pipecat.processors.audio.audio_buffer_processor.time", fake):
        p = await make_processor()
        # Short user burst near t=0. The 16 kHz -> 24 kHz streaming resampler
        # buffers a few frames before it emits any audio, so we send enough
        # frames (8 * 20 ms) to get past that warm-up and actually write the
        # user buffer (which pins _last_user_buffer_update_time). Without a
        # written buffer there is no prior timestamp and the gap-fill cannot
        # fire, so the warm-up frames are required to isolate this mechanism.
        for _ in range(8):
            await send_user(p, burst)
            case_a_gap_fill.clock += FRAME_MS / 1000
        # 700 ms gap with no user frames (e.g. VAD/transport pause at session start).
        case_a_gap_fill.clock += 0.7
        # User resumes.
        for _ in range(8):
            await send_user(p, resume)
            case_a_gap_fill.clock += FRAME_MS / 1000
        user_track, _ = await capture_tracks(p)
        await p.cleanup()

    found = report_case("gap-fill", user_track, RECORD_RATE)
    print()
    return found


async def case_b_bot_sync() -> bool:
    """BUG (b): user silent while the bot speaks its opening line.

    This is the _sync_buffer_to_position path. With no user frames, the bot
    writes ~1.5 s of opening audio. Each bot frame syncs the (empty) user
    buffer up to the bot's position, zero-padding the user/left channel for the
    entire opening. A short user burst at t=0 makes the inserted silence appear
    *after* real audio, matching the customer's "short burst then silence"
    signature.
    """
    print(
        "BUG (b): short user burst, then bot opening while user is silent "
        "(expect: inserted silence via bot-sync)"
    )
    user_burst = make_tone(USER_IN_RATE, FRAME_MS, freq=440)
    bot_frame = make_tone(BOT_OUT_RATE, FRAME_MS, freq=880)

    fake = SimpleNamespace(monotonic=lambda: case_b_bot_sync.clock)
    case_b_bot_sync.clock = 0.0
    with patch("pipecat.processors.audio.audio_buffer_processor.time", fake):
        p = await make_processor()
        # A short real user burst at the very start (~0.00-0.30 s).
        for _ in range(15):  # 15 * 20 ms = 300 ms
            await send_user(p, user_burst)
            case_b_bot_sync.clock += FRAME_MS / 1000
        # Bot speaks its opening line for ~1.5 s while the user stays silent.
        for _ in range(int(1.5 * 1000 / FRAME_MS)):
            await send_bot(p, bot_frame)
            case_b_bot_sync.clock += FRAME_MS / 1000
        user_track, bot_track = await capture_tracks(p)
        await p.cleanup()

    found = report_case("bot-sync", user_track, RECORD_RATE)
    print()
    return found


async def main():
    print("=" * 72)
    print("AudioBufferProcessor startup-silence reproduction")
    print("Pipecat config: input 16 kHz -> record 24 kHz, stereo (user left / bot right)")
    print("=" * 72)
    print()

    control_found = await case_control()
    a_found = await case_a_gap_fill()
    b_found = await case_b_bot_sync()

    print("=" * 72)
    print("SUMMARY")
    print(
        f"  CONTROL (steady audio):       {'silence (UNEXPECTED)' if control_found else 'clean (expected)'}"
    )
    print(
        f"  BUG (a) gap-fill:             {'silence (reproduced)' if a_found else 'clean (NOT reproduced)'}"
    )
    print(
        f"  BUG (b) bot-sync padding:     {'silence (reproduced)' if b_found else 'clean (NOT reproduced)'}"
    )
    print()

    bug_reproduced = (a_found or b_found) and not control_found
    if bug_reproduced:
        print("PASS: the bug reproduced. Leading inserted zero-silence appears in the")
        print("user (left) track in at least one BUG case and is absent in CONTROL.")
    else:
        print("FAIL: the bug did not reproduce as expected.")
        if control_found:
            print("  CONTROL unexpectedly showed leading silence; detection may be too loose.")
        if not (a_found or b_found):
            print("  Neither BUG case produced leading silence; the buffer logic alone may")
            print("  not be the trigger (delivery cadence from Daily may be involved).")

    print("=" * 72)
    return 0 if bug_reproduced else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
