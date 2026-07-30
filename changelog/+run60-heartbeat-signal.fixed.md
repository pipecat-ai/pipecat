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

- Added an `on_heartbeat` event on `PipelineWorker`, fired with each heartbeat's
  traversal latency in seconds. It is the positive counterpart of
  `on_heartbeat_timeout`: consumers that act on missed heartbeats previously had
  no recovery edge and could only decay their state on a wall clock.

- Added `BaseOutputTransport.seconds_since_last_output_write`, a low-false-positive
  watchdog for the one thing heartbeats no longer cover now that they bypass the
  media path: an output audio task that has stopped making progress. Only senders
  with a mixer are considered, since those write every audio chunk whether or not
  anyone is speaking; it returns 0.0 ("no evidence of a wedge") when no sender
  writes continuously.
