- Added a `temperature` field to `HumeTTSService.Settings`, passing Hume's sampling
  temperature through to the synthesis request. Higher values increase variation, lower
  values increase consistency; when unset, Hume applies its own per-model default.
