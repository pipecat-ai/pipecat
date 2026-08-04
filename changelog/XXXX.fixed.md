- Fixed `BaseOutputTransport` hanging pipeline shutdown indefinitely when the
  remote peer stops reading. Transport writes have no timeout of their own, so a
  half-open socket (for example a telephony call already torn down on the
  provider's side) parks the media sender's audio task inside its write. Because
  `process_frame()` pushes the `EndFrame` downstream only after `stop()` returns,
  the `EndFrame` was stranded inside the transport, never reached the sink, and
  `PipelineWorker._wait_for_pipeline_end()` — which applies no timeout on the
  `EndFrame` path — waited on it forever. The drain is now bounded by the new
  `TransportParams.audio_out_drain_timeout_secs` (default 5s), which bounds how
  long the audio task may make *no progress at all* rather than capping the total
  drain, so long queued playout still flushes normally.
