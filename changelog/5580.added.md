- Added a `custom_configuration` field to `NvidiaTTSService.Settings`, which forwards
  model-specific key/value parameters to NVIDIA TTS — Magpie's `flush`,
  `chunk_len_threshold` and `max_chunk_threshold` streaming controls, or Chatterbox's
  `exaggeration_factor`.
