- `OpenAISTTService` now defaults to `gpt-transcribe`, OpenAI's replacement for
  `gpt-4o-transcribe`, which shuts down on 2027-02-26. With
  `include_prob_metrics=True`, GPT transcription models now request `logprobs`
  alongside a `json` response; only Whisper models use `verbose_json`.
