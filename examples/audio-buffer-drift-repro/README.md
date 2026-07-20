# AudioBufferProcessor duration drift reproduction

Minimal, service-free reproduction of a bug in `AudioBufferProcessor` where
recordings grow longer than the real call.

## The bug

`_fill_buffer_silence_gap` (in
`src/pipecat/processors/audio/audio_buffer_processor.py`) injects silence into
a track buffer whenever more than 200 ms of wall-clock time passes between
writes. That is correct when the gap means "no audio existed" (muted mic, idle
bot). But when the gap means "audio existed and was delivered late" (an event
loop stall followed by a catch-up burst of queued frames), the stall window
gets counted twice:

1. The first frame after the stall sees a large elapsed time, so the processor
   injects roughly one stall's worth of silence.
2. The queued frames then burst in back-to-back. Each has a negative gap, and
   there is no trim or reconciliation clause, so each is appended in full.

Net effect: injected silence plus the late real audio both land in the buffer
for the same wall-clock window. Every stall adds about its own duration of
phantom audio, cumulatively. Production reports show recordings ~10% longer
than the real call.

## Files

- `repro.py`: runs a real `Pipeline` (`SimulatedUserMic -> AudioBufferProcessor`)
  via `PipelineWorker`. The mic pushes 20 ms frames at real-time cadence and
  blocks the event loop with `time.sleep()` at scripted points, so queued
  frames burst afterwards, exactly like a transport draining its backlog.
  Prints real audio fed vs wall clock vs recorded duration.
- `eval_duration_drift.py`: asserts recorded duration stays within ±0.25 s of
  the real audio fed in.
- `dump_wav.py`: runs the stall+burst scenario and saves the recorded track to
  a WAV file, so the drift can be heard/inspected directly instead of only
  read as numbers.
- `stall_burst_before_fix.wav` / `stall_burst_after_fix.wav`: the recorded
  track from the stall+burst scenario, captured against the code before and
  after the fix in this PR (see below).

## Run

```bash
cd <pipecat repo root>
.venv/bin/python examples/audio-buffer-drift-repro/repro.py
.venv/bin/python examples/audio-buffer-drift-repro/eval_duration_drift.py
.venv/bin/python examples/audio-buffer-drift-repro/dump_wav.py out.wav
```

## Observed output

Before the fix in this PR (main @ 1.5.1.dev166, commit 5b5cc30ff):

```
scenario                             real audio wall clock   recorded    drift
--------------------------------------------------------------------------------
steady (no stalls)                      10.000s    10.001s    10.000s  +0.000s
2 x 600ms stall + catch-up burst        10.000s    10.001s    11.210s  +1.210s
```

After the fix in this PR:

```
scenario                             real audio wall clock   recorded    drift
--------------------------------------------------------------------------------
steady (no stalls)                      10.000s    10.001s    10.000s  +0.000s
2 x 600ms stall + catch-up burst        10.000s    10.001s    10.000s  +0.000s
```

Before the fix, the stall run records more audio than the entire wall-clock
duration of the call (11.2 s of audio from a 10.0 s call), which is impossible
for a correct recorder. `stall_burst_before_fix.wav` and
`stall_burst_after_fix.wav` are the actual recorded audio from each run.

## Deliberate deviations from a stock bot

- No transport or STT/LLM/TTS services: the simulated mic replaces
  `transport.input()` so the repro is deterministic and free to run. The
  `AudioBufferProcessor` itself is constructed and driven exactly as the docs
  recommend (`auto_start_recording`, `on_track_audio_data`, `stop_recording`).
- The event-loop stall is induced with a blocking `time.sleep()` inside the
  pipeline. That is the point of the repro, not a shortcut: any blocking work
  in a production bot produces the same delivery pattern.
- `pipecat eval` scenarios assert on RTVI conversation events from a live bot;
  they cannot assert recorded-audio duration and cannot deterministically stall
  the bot's event loop. So the eval here applies the duration assertion to the
  same in-process pipeline instead of going through the WebSocket harness.

## Fix

This PR keeps the wall-clock gap fill but reconciles it on catch-up: a small
per-track tracker records the expected buffer position (from elapsed wall-clock
time) and the region of the most recently injected silence. When an appended
frame would push the buffer past the expected position, the overshoot is
trimmed out of that previously injected silence only, never real audio. The
original gap-fill behavior for a genuine gap (muted mic, idle bot, the
"concatenated utterances" bug #4561 that PR #4567 fixed) is unchanged. See
`_SilenceGapTracker` and `_fill_buffer_silence_gap` in
`src/pipecat/processors/audio/audio_buffer_processor.py`.
