- `VADAnalyzer` and `BaseSmartTurn` now shut their per-analyzer
  `ThreadPoolExecutor` down in `cleanup()`. Neither ever did, so each analyzer's
  worker thread (and, for the smart-turn analyzer, the ONNX session it keeps
  alive) was only reclaimed when the executor happened to be garbage collected —
  arbitrarily late for objects held in a pipeline's reference cycles. Shutdown is
  idempotent and non-blocking; analysis after teardown degrades to the last known
  state instead of raising.

- `SileroVADAnalyzer` no longer floods the log on a persistent analysis failure.
  Any exception returns 0 confidence, which is the correct fail-safe but is
  indistinguishable downstream from real silence, so a mismatched sample rate
  presented as a permanently mute caller plus one ERROR per audio frame
  (~50/second/leg). Repeats of the same message are rate-limited and a run of
  consecutive failures is reported once as a degraded analyzer, so the mute has a
  name in the log; recovery is logged too.

- Added `BaseAudioMixer.is_passthrough` (default False). Configuring any mixer
  puts the output transport on a continuous send path — a full-rate
  synthesize/mix/write loop on every leg even while idle, and interruptions that
  drain the audio queue in place instead of cancelling the audio task. That is
  correct for a mixer that generates audio, but not for one that returns its
  input unchanged (e.g. a silence mixer installed when ambient audio is off).
  Mixers that override this to True are treated as if no mixer were configured.
  Default False means no behaviour change for existing consumers.

- Recorded a measured negative result next to the code it concerns
  (`vad_analyzer.py`, `soxr_stream_resampler.py`): per-frame CPU in the VAD,
  smart-turn and resampling paths is two orders of magnitude below the event-loop
  budget and does not explain per-pod concurrency limits. Do not tune it; measure
  scheduling delay via `PipelineWorker`'s new `on_heartbeat` latency instead.
