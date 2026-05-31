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
python -m pipecat.evals evals/01_basic_user_input.yaml
```

Or several at once:

```
python -m pipecat.evals evals/*.yaml
```

## What runs against the sample bot

The sample bot is a passthrough pipeline — input frames go straight to output,
no LLM, no TTS. It only produces user-side events. The first three evals work
against it as-is:

| File | What it shows | Needs Ollama? |
|---|---|---|
| `01_basic_user_input.yaml` | Sanity check — user input produces the expected events | No |
| `02_judge_user_text.yaml` | Judge LLM evaluating natural-language criteria | Yes |
| `03_send_after.yaml` | Event-driven scheduling for chained turns | No |

## Templates for real bots

These evals assert on bot-side events (`llm_response`, `tool_call`,
`bot_stopped_speaking`, `interruption`) and require a bot with real LLM and
TTS services configured. Won't pass against the passthrough sample.

| File | What it shows |
|---|---|
| `04_bot_greeting.yaml` | Bot speaks first; expect-only opening turn |
| `05_tool_call.yaml` | `tool_call` name/args assertions |
| `06_interruption.yaml` | Barge-in via `send_after` on `bot_started_speaking` |

To use these, edit `sample_bot.py` and wire in your LLM and TTS services (the
file has a commented-out example showing the shape).

## Judge configuration

Evals with `says:` / `says_not:` assertions use a judge LLM. The default config
in the YAMLs targets a local Ollama at `http://localhost:11434/v1` with
`llama3:latest`. Change `judge.service` and `judge.model` to use any
pipecat-supported LLM service.
