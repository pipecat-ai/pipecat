- Added `enable_interim_transcription_interruptions` parameter to `LLMUserAggregatorParams` that enables evaluating interruption strategies on interim (partial) transcriptions for faster bot interruption response. This reduces interruption latency by triggering interruptions as soon as the threshold is met, without waiting for the final transcription. Default is `True`.

