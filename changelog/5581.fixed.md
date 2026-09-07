- `TogetherTTSService` now emits sample-aligned audio frames. Together's
  WebSocket audio deltas can end mid-sample, and the odd-length frame that
  resulted crashed the service's audio task when metrics were enabled, leaving
  the turn without audio.
