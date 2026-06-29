# AudioBufferProcessor startup-silence reproduction

A tiny, self-contained harness that reproduces a customer-reported bug at the
frame level: the first 2-3 seconds of a stereo recording sound choppy/garbled
on the user (left) channel because a run of inserted zero-silence appears near
the start of the user track.

No Daily, no network, no API keys. It drives a real `AudioBufferProcessor`
directly with Pipecat frames and prints a clear PASS/FAIL.

## What it reproduces

Customer (Cekura) records calls with `AudioBufferProcessor(num_channels=2,
sample_rate=24000)`, Daily WebRTC transport, user STT at 16 kHz, ElevenLabs TTS
at 24 kHz, on Pipecat 1.3.0. The recording (24 kHz stereo, 16-bit, left = user,
right = bot) shows:

- a short burst of real user audio at the very start (~0.00-0.30 s), then
- a run of EXACT-ZERO samples in the user (left) channel (rms exactly 0.0,
  versus a real captured-quiet noise floor of rms ~0.5 elsewhere), then
- the real conversation, which is clean.

Because the silent stretch is exact zero (not the captured noise floor), it was
INSERTED by `AudioBufferProcessor`, not captured from the mic.

## Implicated source

`src/pipecat/processors/audio/audio_buffer_processor.py` (gap-fill +
buffer-sync behavior, new in Pipecat 1.3.0, PR #4567):

- `_fill_buffer_silence_gap(...)`: when the user buffer was last written more
  than 200 ms ago (`gap > 0.2`), it appends `b"\x00" * silence_bytes`
  proportional to the wall-clock gap. Uses `time.monotonic()`.
- `_sync_buffer_to_position(buffer, target_position)`: pads the lagging track
  with `b"\x00"` so user and bot stay aligned. When the bot writes its opening
  audio while the user is silent, the user buffer gets zero-padded to match.

## The experiment

Three cases, all driven through the public `process_frame` path with a
`StartFrame` first (so the sample rate is set and the task manager is wired),
exactly as `tests/test_audio_buffer_processor.py` does it. The "real" segments
are non-silent sine tones so inserted exact-zero silence is unambiguous. The
recorded tracks are captured via the `on_track_audio_data` event handler, and
the user/left channel is scanned for the longest contiguous run of exact
`0x0000` int16 samples within the first 1.5 s.

- CONTROL: continuous, steady user audio from t=0, interleaved with bot audio.
  Expect no long leading zero run.
- BUG (a) gap-fill: a user burst, then a 700 ms gap with no user frames, then
  resume. Expect a run of exact-zero samples (> 200 ms) near the start.
- BUG (b) bot-sync padding: a short user burst at t=0, then ~1.5 s of bot
  opening audio while the user is silent. Expect the user buffer to be
  zero-padded to align with the bot, producing leading inserted silence.

Config matches the customer exactly:

- `StartFrame(audio_in_sample_rate=16000, audio_out_sample_rate=24000)`
- `AudioBufferProcessor(sample_rate=24000, num_channels=2)`
- user `InputAudioRawFrame` at 16000 Hz (resampled up to 24000 Hz, so the
  input-resample path is exercised)
- bot `OutputAudioRawFrame` at 24000 Hz

### Determinism

Case (a) needs a controllable wall clock. The script monkeypatches the `time`
symbol in the processor module
(`pipecat.processors.audio.audio_buffer_processor.time`), which is what
`_fill_buffer_silence_gap` reads, and advances a fake `monotonic()` manually.
No real sleeps. This is the same patch target used by
`tests/test_audio_buffer_processor.py`.

## Why not the evals framework

The brief suggested trying `pipecat.evals`. That framework is built around
end-to-end scenarios with transports and TTS-generated speech, which is heavier
than needed to exercise the buffer logic in isolation. This repro stays
standalone at the frame level, but borrows the evals/test idioms: `setup()` +
`StartFrame` + `start_recording()` to drive a single processor, and the
time-module monkeypatch from the existing buffer tests.

## How to run

From the repo root:

```bash
uv run python examples/audiobuffer-startup-silence-repro/repro.py
```

(If `uv` is unavailable: create a venv, `pip install -e .` the local pipecat,
then run the same `python ...` command.)

Exit code is 0 on PASS (bug reproduced), 1 on FAIL.

## Expected output

```
CONTROL: continuous user audio from t=0 (expect: clean)
  longest leading zero run: 40 ms (starts at 0.000 s, sample index 0)
  result:                   clean

BUG (a): user burst, >200 ms gap, resume (expect: inserted silence via gap-fill)
  longest leading zero run: 691 ms (starts at 0.138 s, sample index 3300)
  result:                   INSERTED SILENCE DETECTED

BUG (b): short user burst, then bot opening while user is silent ...
  longest leading zero run: 1225 ms (starts at 0.275 s, sample index 6600)
  result:                   INSERTED SILENCE DETECTED

PASS: the bug reproduced. ...
```

(Exact run lengths can vary by a few ms with resampler internals.)

## Notes on the result

- BUG (b) bot-sync is the closest match to the customer's recording: a short
  real user burst followed immediately by a long run of inserted zeros, exactly
  the "burst then silence" signature seen in the .pcm.
- The CONTROL's tiny ~40 ms leading zero run is NOT the bug. It is the 16 kHz
  -> 24 kHz streaming resampler warming up: the first few input frames emit 0
  bytes while the resampler fills its internal buffer, so the merged track
  starts a hair behind the bot track. It is well under the 200 ms threshold and
  is expected.
- That same resampler warm-up makes BUG (b) worse in practice: at session start
  the user resampler emits nothing for the first few frames while the bot's
  24 kHz opening audio (passthrough, no resampling) flows straight into the
  buffer. The user/left channel gets zero-padded to align, which is precisely
  the leading inserted silence the customer hears.
