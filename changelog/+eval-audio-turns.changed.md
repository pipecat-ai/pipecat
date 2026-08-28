- `soundfile` is now a required dependency instead of an optional extra, so
  `SoundfileMixer` and the eval harness can load audio files without any extra
  install step. The `pipecat-ai[soundfile]` extra still resolves and is now a
  no-op, so existing installs keep working unchanged.
