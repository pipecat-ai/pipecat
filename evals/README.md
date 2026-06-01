# Pipecat behavioral evals

YAML files that exercise the eval harness (`pipecat.evals`) against a bot
running with the eval transport (`-t eval`). The eval transport is a plain
WebSocket server speaking [RTVI](../src/pipecat/processors/frameworks/rtvi):
the harness connects as an RTVI client, sends scripted user input, and asserts
on the RTVI events the bot emits.

## Running

In one terminal, start a bot with the eval transport:

```
python examples/voice/voice-cartesia.py -t eval
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

## What makes a bot eval-able

Any bot that runs with `PipelineWorker` already works: `enable_rtvi` defaults
to `True`, so the `RTVIProcessor` and `RTVIObserver` are wired automatically.
The only change a bot needs is an `eval` transport entry, e.g.:

```python
transport_params = {
    "eval": lambda: WebsocketServerParams(
        audio_in_enabled=True,   # for audio-mode evals (STT)
        audio_out_enabled=True,
    ),
    ...
}
```

The runner defaults the serializer to `RTVIEvalSerializer`. `examples/voice/voice-cartesia.py`
is the reference target (real STT + LLM + TTS).

## Two modes: text and audio

Each scenario picks a mode by whether it has a `user_audio:` block.

- **Text mode** (no `user_audio:`): user turns are sent as RTVI `send-text`.
  No VAD/STT runs, so there are **no** `user_started_speaking` /
  `user_stopped_speaking` events — assert on what the bot *did*
  (`llm_started`, `llm_response`, `tool_call`). The top-level `bot_audio:`
  flag (default `true`) maps to `send-text`'s `audio_response`; set it `false`
  to bypass TTS for content-only evals.

- **Audio mode** (`user_audio:` present): the harness renders each user line to
  speech (via the configured TTS, cached on disk) and streams it as RTVI
  `raw-audio`. The bot's VAD/STT run for real, so `user_started_speaking` /
  `user_stopped_speaking` (and `user_transcription`) are genuine signals you
  can assert on.

  ```yaml
  user_audio:
    service: cartesia
    voice: 71a7ad14-091c-4e8e-a314-022ece01c121
    model: sonic-2
    api_key_env: CARTESIA_API_KEY
    sample_rate: 16000
  ```

## Event vocabulary

Scenarios use friendly event names; the harness maps them onto RTVI server
messages:

| scenario `event:`         | RTVI message                                    |
|---------------------------|-------------------------------------------------|
| `user_started_speaking`   | `user-started-speaking` (audio mode)            |
| `user_stopped_speaking`   | `user-stopped-speaking` (audio mode)            |
| `user_transcription`      | `user-transcription` (final), audio mode        |
| `llm_started`             | `bot-llm-started`                               |
| `llm_response`            | `bot-llm-text` accumulated to `bot-llm-stopped` |
| `tool_call`               | `llm-function-call-in-progress`                 |

## The evals

| File                         | Mode  | What it shows                                                    |
|------------------------------|-------|-----------------------------------------------------------------|
| `01_basic_user_input.yaml`   | audio | VAD/STT detect the user's speech (`user_*_speaking`)            |
| `02_judge_bot_response.yaml` | text  | Judge LLM evaluating natural-language criteria on `llm_response` |
| `03_send_after.yaml`         | audio | `send_after` anchored on a real `user_stopped_speaking`         |
| `04_bot_greeting.yaml`       | text  | Bot speaks first; expect-only opening turn; judge on reply      |
| `05_tool_call.yaml`          | text  | `tool_call` name/args assertions                                |
| `06_interruption.yaml`       | text  | Barge-in: a `run_immediately` `send-text` interrupts the bot    |
| `07_stt_in_the_loop.yaml`    | audio | STT-in-the-loop; assert only on `llm_response`                  |

Evals using the `eval:` assertion also need a judge LLM available (Ollama by
default). `05_tool_call.yaml` needs a bot with a `get_weather` tool whose
`RTVIObserver` is configured with `function_call_report_level` `FULL` (the
default `NONE` omits the function name and args).

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
makes sense on `llm_response` — the text the bot produced for this turn. The
parser warns when you put it elsewhere.
