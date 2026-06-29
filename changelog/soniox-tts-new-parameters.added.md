- Added a `speed` setting (0.7-1.3) to `SonioxTTSService`.

- `SonioxTTSService` now supports cloned voices: pass the voice UUID as `voice`.

- `SonioxTTSService` now emits word-aligned `TTSTextFrame`s (disable with `return_timestamps=False`), so context reflects only what was spoken on interruption.
