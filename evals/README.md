# Pipecat behavioral evals

YAML files that exercise the eval harness (`pipecat.evals`) against a bot
running with the eval transport (`-t eval`).

## Running

In one terminal, start the sample bot:

```
python evals/sample_bot.py -t eval
```

In another terminal, run an eval:

```
python -m pipecat.evals run evals/01_basic_user_input.yaml
```

Or several at once:

```
python -m pipecat.evals run evals/*.yaml
```

Or via pipecat-cli (when installed):

```
pipecat eval run evals/*.yaml
```

## What runs against the sample bot

The sample bot is a passthrough pipeline — input frames go straight to output,
no LLM, no TTS. It only produces user-side events. Two evals work against it
as-is:

| File | What it shows |
|---|---|
| `01_basic_user_input.yaml` | Sanity check — user input produces the expected events |
| `03_send_after.yaml` | Event-driven scheduling for chained turns |

## Templates for real bots

These evals assert on bot-side events (`llm_started`, `llm_response`,
`tool_call`, `interruption`) and require a bot with a real LLM configured.
Won't pass against the passthrough sample. Evals using the `eval:`
assertion also need a judge LLM available (Ollama by default).

| File | What it shows |
|---|---|
| `02_judge_bot_response.yaml` | Judge LLM evaluating natural-language criteria on `llm_response.text` |
| `04_bot_greeting.yaml` | Bot speaks first; expect-only opening turn; judge on bot reply |
| `05_tool_call.yaml` | `tool_call` name/args assertions |
| `06_interruption.yaml` | Barge-in via `send_after` on `llm_started` |

To use these, edit `sample_bot.py` and wire in your LLM and TTS services (the
file has a commented-out example showing the shape).

## Judge configuration

Evals with `eval:` assertions use a judge LLM. Configure with a top-level
`judge:` block:

```yaml
judge:
  service: ollama             # default
  model: qwen2.5:3b           # default
  endpoint: http://...        # optional, service-specific default if omitted
```

The judge runs as a one-shot call through any pipecat LLM service that exposes
`run_inference()` (currently Ollama and OpenAI). The `eval:` assertion only
makes sense on `llm_response` — the text the bot produced for this turn.
Asserting on user transcripts is silly since the test controls what the user
said; the parser warns when you do it.

## Modes: real-time vs fast

The transport supports two pacing modes, chosen via top-level `fast:` in the
YAML (default `false`):

- **Real-time** (`fast: false`): pacing happens on each `TextFrame` chunk,
  sleeping for `len(text) / chars_per_second` seconds (≈ 150 WPM by
  default). Gives interruption tests a realistic window in which to barge
  in. Use for `06_interruption.yaml` and similar timing-sensitive evals.
- **Fast** (`fast: true`): the transport pushes
  `LLMConfigureOutputFrame(skip_tts=True)` so the LLM produces text without
  invoking TTS at all — no audio, no TTS API calls, no pacing. Use for
  text-only evals where interruption timing doesn't matter and you want
  results as fast as possible.
