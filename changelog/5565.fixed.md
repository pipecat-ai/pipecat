- `SambaNovaLLMService` now sends the `frequency_penalty`, `presence_penalty` and
  `seed` settings, which SambaNova's chat completions API accepts. They were
  previously dropped from the request.
