- `SarvamLLMService` now defaults to `sarvam-105b`, the only chat model Sarvam's
  API still serves. `sarvam-30b`, `sarvam-30b-16k` and `sarvam-105b-32k` are
  rejected server-side, so they're no longer accepted as the `model` setting.
