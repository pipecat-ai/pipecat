# Release Evals

Before a Pipecat release we make sure the examples still work. There are more
than a hundred of them, so doing it by hand is slow and painful. The release
evals run each example for us.

## How it works

Each example is a Pipecat **bot**. We run it with its eval transport
(`-t eval`). The **eval harness** (`pipecat.evals`) connects to it as an RTVI
client, plays the user's side of a conversation, listens to what the bot says,
and judges it.

A **scenario** is one check: a conversation to have with the bot and how to
decide whether the bot did well. A YAML file holds one or more scenarios under
`scenarios:`. Each one runs on its own, against its own bot, and is named
`<file>/<scenario>`. There are two kinds:

- A **scripted** scenario (`scenarios/scripted/<name>.yaml`) writes the user's
  turns out, with what we expect from the bot after each one. For example,
  `capital_question` asks "What is the capital of Germany?" and checks that the
  reply says Berlin. See [Scripted scenarios](#scripted-scenarios).
- A **simulated** scenario (`scenarios/simulated/<name>.yaml`) lets an LLM play
  a caller with a goal. A judge then reads the whole conversation and decides
  whether the bot did its job. See [Simulations](#simulations).

Scenarios are shared, so one scenario can check many bots.
[`manifest.yaml`](manifest.yaml) says which scenarios each bot runs.

## What you need

The harness runs the judge, the user's voice and the transcriber for the bot's
speech, so a few things need to be in place.

**The judge.** The scenarios judge with [Jev](https://typesafe.ai), a hosted
classifier from TypeSafe. It answers each check in a few hundred milliseconds
and says how sure it is. The judge is picked in `judge_text.yaml`,
`judge_audio.yaml` and the audio scenario of `language_switch.yaml` through a
factory we keep in `evals/judges.py` (`factory: evals.judges.typesafe_classifier`).
It needs the `jev` extra and `TYPESAFE_API_KEY` exported in the shell that runs
the suite. The harness doesn't read `.env`; only the bots do.

Jev gives verdicts but not reasons. So a local LLM, the **explainer**, writes
the reason for every `no` and for every verdict Jev is not sure about. Jev's
verdict always stands.

**A local LLM.** The explainer, and the caller in simulations, run `gemma4:12b`
on [Ollama](https://ollama.com). Install Ollama, start it, and pull the model:

```sh
ollama pull gemma4:12b
```

Its config sets `reasoning_effort: none`. `gemma4` can think before answering,
but we only read its JSON answer, so thinking just makes each call slower.

**Local audio models** (audio scenarios only). The user's voice is synthesized
with Kokoro and the bot's speech is transcribed with
[Moonshine](https://github.com/moonshine-ai/moonshine). Both run from local
ONNX files that download once into `~/.cache/pipecat/`. Synthesized user turns
are cached under `~/.cache/pipecat/evals/tts`, so a repeated scenario doesn't
synthesize them again. No keys, no cost per run. A scenario can pick another
service in its `user.speech:` and `judge.transcription:` blocks, such as
Whisper. The defaults are English only, so `language_switch/audio` uses
Whisper's `tiny` model (75MB).

**Node.js** (MCP bot only). `mcp/mcp-stdio.py` starts its MCP server with
`npx`. The package downloads on first use.

**Each bot's own keys.** A bot is a real example, so it needs the API keys it
always needs, in your `.env` (`OPENAI_API_KEY`, `CARTESIA_API_KEY`,
`DEEPGRAM_API_KEY`, ...). A bot with missing keys fails its eval.

Install everything with the extras:

```sh
uv sync --group dev --all-extras --no-extra gstreamer --no-extra local
```

### Judging with Ollama alone

You can judge without Jev, with no key and no network for the judge. Replace
the `eval:` block in `judge_text.yaml`, `judge_audio.yaml` and the audio
scenario of `language_switch.yaml` with the explainer's own block
(`service: ollama`, `model: gemma4:12b` and its `extra:`).

The local judge is then called once per `eval:` in a scripted scenario and once
per bot turn in a simulation, and every run shares one copy of it. So its speed
sets the pace of the whole suite. It has to be fast, accurate, and give the same
verdict on the same input. `gemma4:12b` answers in under a second and is stable.
Smaller models are fast too, but they misread short interim replies. A bot that
has only said "Let me check on that." should get `continue` (wait for the rest),
and a `yes` there passes a turn in which the bot said nothing. Older models also
fail correct spoken answers that the transcriber turned into a homophone, like
"four" heard as "for".

A `judge.eval:` block can point at any other classifier or LLM with a
`factory:`, a dotted path to a callable that takes the block and returns a
`BaseClassifier` or an OpenAI-compatible LLM service. `evals/judges.py` is one.

## Running

Export the Jev key first, then run the suite:

```sh
export TYPESAFE_API_KEY=...
./run.sh                      # everything in the manifest
./run.sh -p voice-openai      # only bots whose path contains "voice-openai"
./run.sh -s capital_question  # only the capital_question scenario
./run.sh -c 8                 # 8 at a time
./run.sh -n nightly           # write to test-runs/nightly/ instead of a timestamp
```

`run.sh` is `pipecat eval suite` with `-d`, so the full debug logs are always
saved. Any other flag goes through:

```sh
uv run python -m pipecat.evals suite -d manifest.yaml [-p PATTERN] [-s SCENARIO] [-c N] [-n NAME] [-t SECS] [-a] [--no-cache] [--repeat N]
```

Each run writes to `test-runs/<name>/`, a timestamp unless you pass `-n`:

- `logs/<bot>__<scenario>.log`: the bot's output.
- `logs/<bot>__<scenario>.eval.log`: what the harness decided and why. Start
  here when a run fails.
- `logs/<bot>__<scenario>.debug.log`: the full logs of every pipeline the
  harness ran (user speech, transcription, judge, harness).
- `recordings/<bot>__<scenario>.wav`: the conversation audio of an audio
  scenario. The manifest sets `record: true`; `-a` forces it on.

Other flags: `-c` for concurrency, `-t` for the default timeout of an
expectation without its own `within_ms`, and `--no-cache` to synthesize the
user's audio again instead of reusing it. The suite keeps `concurrency` runs
going, taking the next one from the manifest entries in turn. An entry whose
provider rate-limits sets its own `concurrency:`, and never has more than that
many runs going at once.

Everything in the manifest header except `suite:` can be given on the command
line instead, and the command line wins: `--bots-dir`, `--scenarios-dir`,
`--runs-dir`, `--base-port`, `--cache-dir`, `--spawn`, `--python`.

### Repeating a run

One run says whether a bot passed. `--repeat N` says how often it passes. That
is the question that matters for anything with a race in it, like
interruptions, async function results or turn detection, where a bot can pass
half the time and look fine in any one run.

```sh
./run.sh -p function-calling -s async_tool/delivery --repeat 50 -c 3
```

Attempts run one after another across all bots (`A#1, B#1, C#1, A#2, ...`),
so every bot sees the same machine conditions. A slow moment shows up in all
of them at once, not as a regression in one bot. Each attempt adds its number
to its file names (`..._001.log`, `..._002.log`).

The result is a pass rate per bot and scenario. Failures are grouped by kind
(`timeout`, `judge_no`, `missing_function_call`, ...; see `FAILURE_KINDS` in
`pipecat.evals.results`):

```
  Failures (35 of 150):
     10x  turn 3  response       timeout                 google 4, openai-async 4, anthropic 2
      7x  turn 3  response       judge_no                google 4, openai-responses 3
      3x  turn 1  function_call  missing_function_call   anthropic 2, openai-async 1
```

A repeated run always exits 0. It reports a rate; what rate is acceptable is
up to you.

Every run, repeated or not, also writes `results.jsonl`, one JSON line per run
as it finishes. Each line has the kind of run (`script` or `simulation`), the
outcome, the failures with their kind, each turn's status and what its
expectations matched, and the paths to its files. Runs that failed also carry
`events_seen`, a record of what the bot did, which is usually where the cause
is.

### GPU and concurrency

Only the local LLM runs on the GPU. Ollama keeps one copy of `gemma4:12b`
loaded (about 9GB, mostly its context window), so GPU use stays about the same
whatever `-c` is. Kokoro and Moonshine run on the CPU, so concurrency is bound
by CPU and RAM. A 16GB GPU runs the default setup with room to spare. A much
larger model is what would run out of memory, and that shows up as a harness
error in the run's `.eval.log`. On a smaller card, `num_ctx` in the model's
`extra:` block trims the context; it never needs more than a few thousand
tokens.

Whisper is an alternative transcriber (`transcription: {service: whisper}`). It
runs on the CPU by default and takes `device: cuda` if you have room.

## One scenario against a running bot

If a bot is already running with `-t eval`, run a scenario against it
directly. This is handy while working on a scenario or a bot:

```sh
pipecat eval run scenarios/scripted/capital_question.yaml --bot-url ws://localhost:7860
pipecat eval run scenarios/simulated/capital_curious.yaml --bot-url ws://localhost:7860 -v
```

## Scripted scenarios

A scripted scenario is a list of `turns`. A turn sends a `user` utterance,
presses DTMF keys with `dtmf:`, or sends nothing and only checks what the bot
does, which is how a bot's opening greeting is tested. The full format
(events, expectations, `send_after:`, `image:`, ...) is in the
[`pipecat.evals.script`](../../src/pipecat/evals/script.py) module docstring.

A few things to know when writing one:

- **Several scenarios per file.** A file's `scenarios:` list can hold many
  short conversations that test one behavior, like the turn-completion cases.
  Any scenario key can also sit at the top of the file as the default for all
  of them. A scenario that sets the same key replaces it whole, so a `context:`
  is always written out in full. `turns:` at the top with one entry per judge
  or modality runs the same conversation under each (`interruption`,
  `capital_curious`). `persona:` at the top with a `goal:` per entry sends the
  same caller on different errands. `-s <file>` selects them all and
  `-s <file>/<scenario>` one.
- **Modality.** The `judge:` and `user:` blocks pick audio or text. In audio
  mode the user's turns are synthesized, so the bot's STT is really used, and
  the judge reads a transcription of the bot's audio. Text mode sends and judges
  text, and is faster and silent.
- **Recordings.** In audio mode a turn can play a file instead of synthesizing:
  `audio: ../assets/<clip>.wav` streams it to the bot, and `user:` says what it
  contains, for the judge and `text_contains`. Recordings live in `assets/`.
- **Greet first.** Most bots greet on connect. That greeting has to finish
  before the first user turn, or the question barges into it. So user-first
  scenarios start with a bot-first turn that expects the greeting.

Shared `judge:` and `user:` blocks live in small files in `scenarios/`
(`judge_audio.yaml`, `judge_text.yaml`, `user_audio.yaml`, and
`simulator.yaml` for simulations). Scenarios pull them in with `!include`,
relative to the scenario file:

```yaml
user: !include ../user_audio.yaml
judge: !include ../judge_audio.yaml
```

### Vision

Some bots need data they would normally get from a `/start` request, such as
a vision bot's image. The eval transport has no such request, so a bot entry
gives a `runner_body:` instead, either a file (relative to the manifest) or
written inline:

```yaml
- bot: vision/vision-openai.py
  runner_body:
    path: scenarios/vision-cat.yaml   # image_path: ../assets/cat.jpg, question: ...
  scenarios: [vision_describe]
- bot: turns/filter-incomplete-turns.py
  runner_body:
    data: {model: gpt-4o-mini}
  scenarios: [turn_completion]
```

A bot given a file starts in that file's directory, so a relative `image_path`
in the body resolves next to it. Several entries can share one bot and differ
only in their body, for example to sweep models. Give each a `name:`
(`name: groq/llama-3.3-70b`) so the output, `-p`, `results.jsonl` and the log
names tell them apart. `vision_describe` is a bot-first turn: the bot describes
the image on connect and the judge checks that it saw a cat.

A turn can also register an `image:` that the eval transport serves when the
bot asks for a user image mid-conversation (see `describe_image`).

### Flows

The `flows/` bots have their own scenarios. They check which functions fire,
with which arguments, and what the bot says back. Each scenario targets one
feature of its example: dynamic routing, direct and global functions,
`FlowsFunctionSchema` constraints, context strategies, conditional branching,
multi-worker handoff, `LLMSwitcher`.

They run in text mode. To drive a bot's real audio pipeline, add the shared
includes (`user: !include user_audio.yaml`, `judge: !include judge_audio.yaml`).

When writing one:

- Each turn checks the `function_call` and a `response` eval. The `response`
  event also paces the run: the harness waits for the bot to finish before the
  next turn.
- The last turn checks only the function call. `end_conversation` tears the
  pipeline down before the goodbye reaches the harness.

The bots pick their LLM from `$LLM_PROVIDER` (default `openai_responses`;
`hello_world` always uses Google): `LLM_PROVIDER=anthropic ./run.sh -p flows`,
also `google` and `aws`. `llm_switching` needs OpenAI, Google and Anthropic
keys. `warm_transfer.py` needs Daily and a live human agent, so it isn't
covered.

## Simulations

A scripted scenario writes the user's side out. A **simulation** replaces it
with a **persona**: an LLM playing a caller with a goal. It says whatever the
conversation calls for and hangs up (an `end_call` tool) when the goal is
reached or clearly out of reach. A judge then reads the whole conversation,
with the tools the bot called, and decides whether the caller got what they
came for. A plain voice bot takes a curious caller in text and in audio, which
checks the simulation itself in both modes. The rest cover the Flows examples,
because those bots have a job to finish: book a table, take a patient's intake,
place an order, quote a policy.

```sh
./run.sh -k simulation             # every simulation, nothing scripted
./run.sh -p flows                  # the Flows bots, scripted and simulated
./run.sh -s book_table/available   # one simulation, as many runs as its file says
./run.sh -s order_pizza -r 5       # one simulation, five runs
```

A simulation runs once unless its file says otherwise (`runs`) or `-r` repeats
it, and every run must pass. A persona never says the same thing twice, so
repeat a doubtful result instead of trusting one run. A run passes when the
judge says the bot did its job (`success`), no judged metric scored below its
`min_score`, and no measured metric failed. A failed run says which one gave
way.

The judge sees the bot's tool calls, name and arguments, but not their results.
Whether the bot made a call at all is a `function_calls` measure, no judge
needed. If a reply must match backend data, write the expected value into
`success` or the criterion ("the reply says the appointment is on Tuesday
September fifteenth") and keep the mocks deterministic.

A judged metric's `criterion` says what every reply should be. The judge
decides it for each bot turn with a yes or a no, and the score is the share of
turns that got a yes. `min_score: 1` means always, and `0.8` allows one slip in
five. Write a rule as a condition and say what a reply outside it does ("when
the reply turns down a time, it offers alternatives; a reply that turns down no
time passes"), or the judge reads a "never" as an "always". Something the bot
must do once belongs in `success`.

A measured metric (`measure: turns`, `duration`, `words` or `latency`, with
`min_value` and `max_value`) is computed from the run. The per-reply ones bound
every reply, so `latency` is the slowest reply and `words` the longest.
`measure: function_calls` takes a `calls:` list instead: the calls the bot
should make, by name and optionally `args`, and fails on a missing or an extra
call. `calls: []` means the bot must call nothing, which is how you check a
caller who should be turned down.

`results.jsonl` carries each metric's score and every turn's verdict. The suite
prints a pass rate per simulation and exits non-zero when a run failed.
`--repeat` turns it into a measurement, with rates and exit code 0. A run that
errored (the bot never came up, the persona's LLM failed, the judge gave no
verdict) is reported but left out of the rate. A run where nobody says anything
for `max_silence_s` (30 s by default) ends as `silence`.

| Simulation                | Bot                                              | The caller                                                       |
| ------------------------- | ------------------------------------------------ | ---------------------------------------------------------------- |
| `capital_curious/text`    | `voice/voice-cartesia.py`                        | Asks the capital of Germany and hangs up, in text.               |
| `capital_curious/audio`   | `voice/voice-cartesia.py`                        | The same caller, speaking and listening.                         |
| `book_table/available`    | `flows/restaurant_reservation.py`                | Books a table for two at 6 PM, which is free.                    |
| `book_table/flexible`     | `flows/restaurant_reservation.py`                | Wants 7 PM (taken) for four but accepts anything from 6 to 9 PM. |
| `book_table/impossible`   | `flows/restaurant_reservation.py`                | Can only do 7 or 8 PM, both taken; success is a graceful no.     |
| `complete_patient_intake` | `flows/patient_intake.py`                        | Gives a birthday, a prescription, an allergy, and a condition.   |
| `order_pizza`             | `flows/food_ordering.py`                         | Orders a large pepperoni pizza and asks about delivery time.     |
| `order_sushi`             | `flows/food_ordering_advanced_functionschema.py` | Orders three California rolls.                                   |
| `get_insurance_quote`     | `flows/insurance_quote.py`                       | Gets a quote, then a second one with more coverage.              |

The persona LLM is the `simulator:` block, the same local Ollama model by
default, so a simulation needs no API key. `simulator.yaml` is where to point
every simulation at another model. In audio mode the persona's turns are
synthesized and the bot's speech transcribed with the same services as a
scripted audio scenario, so `capital_curious/audio` exercises the bot's STT,
TTS and turn taking against a caller of its own. The file format is in the
[`pipecat.evals.simulation`](../../src/pipecat/evals/simulation.py) module
docstring. Run one by hand with `pipecat eval run
scenarios/simulated/<name>.yaml --bot-url ws://localhost:7860 -v`, which prints
the conversation as it happens.

## Adding coverage

- New bot: add an entry to `manifest.yaml` with `bot:` and the `scenarios:` it
  should run.
- New behavior to test: add `scenarios/scripted/<name>.yaml` and list it in the
  manifest as `scripted/<name>`. Several short cases of one behavior go in one
  file.
- New goal to reach: add `scenarios/simulated/<name>.yaml` with a `persona:`
  and list it as `simulated/<name>` under the bot that serves it.
