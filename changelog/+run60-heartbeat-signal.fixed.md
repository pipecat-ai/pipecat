- Fixed the pipeline heartbeat so it measures pipeline health instead of
  bot-audio backlog and barge-in cadence. `HeartbeatFrame` was a plain
  `ControlFrame`, so `BaseOutputTransport` routed it into the media sender's
  audio queue — which is consumed at 1x realtime — making the measured
  traversal latency at least as long as the bot audio already queued ahead of
  it (a single ~9s utterance was enough to trip a 10s
  `heartbeats_monitor_secs`). It was also interruptible, so every processor's
  `FrameQueue.reset()` on an `InterruptionFrame` destroyed the heartbeats in
  flight; under continuous barge-in a perfectly healthy pipeline could go
  minutes without a heartbeat. Heartbeats now bypass the media senders and are
  pushed straight to the sink, and `HeartbeatFrame` is an `UninterruptibleFrame`
  (explicitly excluded from the "don't cancel the process task" guard in
  `_start_interruption`, so it can never wedge a processor).

- Heartbeats are also exempt from `FrameProcessor.pause_processing_frames()`.
  Every TTS service constructed with `pause_frame_processing=True` (ElevenLabs,
  Deepgram, Rime, Azure, Fish, Groq, LMNT, Inworld, Neuphonic, Sarvam, …) pauses
  from `LLMFullResponseEndFrame` until `BotStoppedSpeakingFrame`, i.e. for an
  entire utterance playout, so without this the same "traversal latency equals
  playout backlog" defect simply moved from the output transport to the TTS
  processor. The pause still blocks every other frame, so the ordering of real
  work is unchanged.

- Added an `on_heartbeat` event on `PipelineWorker`, fired with each heartbeat's
  traversal latency in seconds. It is the positive counterpart of
  `on_heartbeat_timeout`: consumers that act on missed heartbeats previously had
  no recovery edge and could only decay their state on a wall clock. A blocked
  pipeline now accumulates heartbeats rather than having them purged, so the
  monitor coalesces a drained backlog into a single event carrying the newest
  measurement — otherwise one momentary drain would fire a burst of stale
  "recovered" edges. Handlers must still ignore a latency above
  `heartbeats_monitor_secs`, and must read the value as scheduling delay plus the
  longest in-flight per-processor operation (a streamed LLM generation still sits
  in front of a heartbeat), not as pure event-loop delay.

- Added `BaseOutputTransport.seconds_since_last_output_write`, a low-false-positive
  watchdog for the one thing heartbeats no longer cover now that they bypass the
  media path: an output audio task that has stopped making progress. Only senders
  with an audio-generating mixer are considered, since those write every audio
  chunk whether or not anyone is speaking; it returns 0.0 ("no evidence of a
  wedge") when no sender writes continuously. Note the corresponding coverage
  hole: a sender without such a mixer now has no output-task liveness signal at
  all and needs a different check.
