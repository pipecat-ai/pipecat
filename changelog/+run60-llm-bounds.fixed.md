- Bounded the `AsyncOpenAI` client built by `BaseOpenAILLMService.create_client`.
  It previously passed neither `max_retries` nor `timeout`, so SDK defaults
  (`max_retries=2`, `timeout=600s`) applied and `retry_on_timeout` defaults to
  False — a black-holed endpoint could hold a live conversational turn open for
  half an hour. Now defaults to `CLIENT_MAX_RETRIES=1` and an `httpx.Timeout` of
  5s connect / 30s read / 10s write / 5s pool, both overridable per service via
  the new `client_max_retries` and `client_timeout` constructor arguments. This
  covers every OpenAI-compatible provider that inherits `create_client`
  (Speaches/vLLM, Groq, Together, Fireworks, OpenRouter, xAI, …); providers that
  build their own client (Azure, Google, AWS) are unaffected.

- `BaseOpenAILLMService.run_inference` now carries its own per-request timeout
  (`INFERENCE_TIMEOUT_SECS`, 180s, overridable via the new
  `inference_timeout_secs` constructor argument) instead of inheriting the
  client's conversational read bound. Out-of-band inference is non-streaming, so
  the server sends nothing until the whole completion is generated and the read
  timeout applies to total generation rather than time-to-first-byte; the 30s
  that is right for a first token on a live turn would silently cap post-call
  analysis (transcript summaries, slot extraction, tagging, classification).

- Added `EMPTY_RETRY_TOTAL_BUDGET_SECS` (6s), a wall-clock budget covering all
  generation attempts of a single turn. The empty-completion recursion, the
  `APIConnectionError` reconnect and the SDK's own retries previously multiplied
  with no aggregate bound. A budget exhaustion logs distinctly from the depth
  cap so its rate is measurable before the value is tuned.

- Bounded `ReasoningTagGate` hold mode with `REASONING_HOLD_MAX_CHARS` (240).
  A lone reasoning opener with no closer — Gemma-4's no-thinking-mode artifact —
  used to withhold the entire reply until `flush()`, so downstream TTS received
  nothing until the generation finished and time-to-first-audio became full
  generation time. Past the bound, a held block with no closer and no
  chain-of-thought signature is released and the gate starts streaming; a closer
  arriving later is still swallowed rather than spoken. Genuine chain-of-thought
  (which carries the signature) still holds to end of stream. Added
  `reasoning_suppressed_chars_total` and `reasoning_bounded_releases_total`
  counters on the service so gate activity is queryable, not just greppable.
