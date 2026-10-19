# LLMWithBackend tuning results

The latest results of the [tuning checks](README.md), from the last change to `LLMWithBackend`'s prompts or classifier. Each update is committed with the change it measures, so `git log -p` on this file ties every number to its commit and shows the numbers before it.

## Classifier replay

Predictive delegation's classifier on the recorded turns in `classifier_replay/`, with `jev-1.13.0` (2026-10-14):

```bash
uv run python evals/llm-with-backend/tuning/classifier_replay/replay.py
```

```
107 turns, threshold 0.6
  false "yes" (delegated early, not needed): 0/59
  missed "yes" (left to the frontend):       0/48
  latency: p50 139ms, p90 201ms

probability of a "yes" per utterance: median (min-max)
  no  0.17 (0.14-0.19)  What is a race condition in programming?
  yes 0.96 (0.96-0.97)  Can you fix the flaky retry test in the HTTP client for me?
  no  0.13 (0.12-0.15)  While we wait, tell me a quick joke.
  no  0.37 (0.33-0.56)  How's it going with the fix?
  yes 0.94 (0.92-0.95)  Also, what's the CI status on main?
  no  0.19 (0.15-0.27)  What's the capital of Australia?
  yes 0.95 (0.93-0.95)  Can you find our internal docs on the deploy process?
  yes 0.94 (0.92-0.95)  Actually, never mind the docs.
  no  0.25 (0.19-0.30)  Thanks. What year was the first moon landing?
```

## Text evals

The twelve text scenarios on the three bots with a text frontend, five runs each (2026-10-14). The frontends run on their services' default models (`gpt-4.1`, `gemini-3.6-flash`, `claude-sonnet-4-6`), the backend on `claude-sonnet-4-6`, and the judge is the release suite's text judge. Each table comes from the run's `results.jsonl` through `results_table.py`:

```bash
uv run python -m pipecat.evals suite evals/llm-with-backend/manifest.yaml -s text --repeat 5
uv run python evals/llm-with-backend/tuning/results_table.py <runs-dir>/<run>/results.jsonl
```

### Without predictive delegation

| scenario | anthropic-frontend | google-frontend | openai-responses-frontend |
| --- | --- | --- | --- |
| `accepted_offer/text` | 5/5 | 5/5 | 5/5 |
| `aside_while_waiting/text` | 5/5 | 5/5 | 5/5 |
| `cancel_and_replace/text` | 5/5 | 5/5 | 5/5 |
| `cancel_delegation/text` | 4/5 | 5/5 | 1/5 |
| `concurrent_delegations/text` | 0/5 | 1/5 | 1/5 |
| `delegate_ack/text` | 5/5 | 5/5 | 5/5 |
| `delegate_with_aside/text` | 5/5 | 5/5 | 5/5 |
| `frontend_only/text` | 5/5 | 5/5 | 5/5 |
| `implicit_cancel/text` | 1/5 | 5/5 | 2/5 |
| `partial_cancel/text` | 5/5 | 5/5 | 0/5 |
| `progress_inquiry/text` | 5/5 | 5/5 | 3/5 |
| `superseding_delegation/text` | 5/5 | 5/5 | 2/5 |
| **total** | **50/60** | **56/60** | **39/60** |

### With predictive delegation

The same, with `predictive_delegation=PredictiveDelegation(JevClassifier(...))` added to each bot's `LLMWithBackend`.

| scenario | anthropic-frontend | google-frontend | openai-responses-frontend |
| --- | --- | --- | --- |
| `accepted_offer/text` | 5/5 | 5/5 | 5/5 |
| `aside_while_waiting/text` | 5/5 | 4/5 | 5/5 |
| `cancel_and_replace/text` | 5/5 | 5/5 | 5/5 |
| `cancel_delegation/text` | 4/5 | 5/5 | 5/5 |
| `concurrent_delegations/text` | 0/5 | 0/5 | 1/5 |
| `delegate_ack/text` | 5/5 | 5/5 | 5/5 |
| `delegate_with_aside/text` | 5/5 | 5/5 | 5/5 |
| `frontend_only/text` | 5/5 | 5/5 | 5/5 |
| `implicit_cancel/text` | 1/5 | 5/5 | 5/5 |
| `partial_cancel/text` | 5/5 | 5/5 | 5/5 |
| `progress_inquiry/text` | 5/5 | 5/5 | 2/5 |
| `superseding_delegation/text` | 5/5 | 5/5 | 5/5 |
| **total** | **50/60** | **54/60** | **53/60** |

### Known failures

- `concurrent_delegations` fails on every frontend most of the time: the judge rejects the reply that reports both results.
- gpt-4.1 often answers a stop itself, with no `delegate` call, so the backend carries on (`cancel_delegation`, `implicit_cancel`, `partial_cancel`, `superseding_delegation`). With predictive delegation, the classifier delegates those stops early, and they pass.
- The other failures are the judge answering "continue" to a correct one-line acknowledgement.

## Audio evals

The thirteen audio scenarios on all five bots, one run each, without predictive delegation (2026-10-14). The user's voice is synthesized with Kokoro, the bot's speech transcribed with Moonshine, and the judge is the release suite's audio judge.

```bash
uv run python -m pipecat.evals suite evals/llm-with-backend/manifest.yaml -s audio
```

| scenario | anthropic-frontend | gemini-live-frontend | google-frontend | openai-realtime-frontend | openai-responses-frontend |
| --- | --- | --- | --- | --- | --- |
| `accepted_offer/audio` | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| `aside_while_waiting/audio` | 0/1 | 1/1 | 0/1 | 1/1 | 1/1 |
| `cancel_and_replace/audio` | 1/1 | 0/1 | 1/1 | 0/1 | 1/1 |
| `cancel_delegation/audio` | 0/1 | 1/1 | 1/1 | 0/1 | 0/1 |
| `concurrent_delegations/audio` | 0/1 | 0/1 | 0/1 | 1/1 | 0/1 |
| `delegate_ack/audio` | 1/1 | 0/1 | 1/1 | 1/1 | 1/1 |
| `delegate_with_aside/audio` | 1/1 | 1/1 | 0/1 | 1/1 | 0/1 |
| `frontend_only/audio` | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| `implicit_cancel/audio` | 0/1 | 1/1 | 1/1 | 0/1 | 0/1 |
| `partial_cancel/audio` | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| `progress_inquiry/audio` | 0/1 | 0/1 | 1/1 | 0/1 | 1/1 |
| `result_during_interruption/audio` | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| `superseding_delegation/audio` | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| **total** | **8/13** | **9/13** | **10/13** | **9/13** | **9/13** |

One run per scenario is noisy: the same pass on the code before this change scored 54/65. The failures here are the judge hesitating on acknowledgements, the speech-to-speech bots' `cancel_and_replace` race, and one gpt-4.1 `delegate_with_aside` run that promised the fix without calling `delegate`. Run five times on each bot, `delegate_with_aside` passed 19/25 here against 14/25 before, with one more gpt-4.1 run that skipped the call (none before).
