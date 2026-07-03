# Release Evals

Before a Pipecat release we make sure all (or most) of the 100+ examples still
work. Doing that by hand is slow and painful, so these "release evals" drive each
example automatically.

## How it works

Each example is a Pipecat **bot**. We run it with its eval transport
(`-t eval`), and the **eval harness** (`pipecat.evals`) connects to it as an RTVI
client, plays the user's turns (synthesizing audio when a scenario is in audio
mode), transcribes the bot's speech, and judges the response with an LLM.

A scenario (`scenarios/<name>.yaml`) is a scripted conversation plus the
expected results. For example the `capital_question` scenario asks "What is the
capital of Germany?" and judges that the reply says Berlin. Scenarios are
reusable, so one shared scenario covers many bots.

[`manifest.yaml`](manifest.yaml) maps each bot to the scenarios it runs.

## Prerequisites

The harness runs the judge, the user's voice, and the bot-speech transcriber
*locally* by default, so you need a few things in place:

- **A judge LLM.** Scenarios judge with [Ollama](https://ollama.com) by default
  (`http://localhost:11434`). Install Ollama, start it, and pull the model the
  scenarios use: `ollama pull gemma2:9b`. We use `gemma2:9b` because smaller
  judges (e.g. `llama3:latest`, `qwen2.5:7b`) too often reject a correct spoken
  answer the transcriber mangled into a homophone — "four" heard as "for" — and
  it still fits a 16GB GPU alongside the audio models. (A scenario's `judge:`
  block can point at OpenAI instead — set `service: openai` and `$OPENAI_API_KEY`.)
- **Local audio models** (audio-mode scenarios only). The user's voice is
  synthesized with Kokoro TTS and the bot's speech is transcribed with
  [Moonshine](https://github.com/moonshine-ai/moonshine) (Whisper is available
  as an alternative via the scenario's `transcription:` block). All run from
  local ONNX/model files that download once on first use (cached under
  `~/.cache/pipecat/evals/tts`). No keys, no per-run cost.
- **Each bot's own credentials.** A bot is a real example, so it needs the same
  service API keys it normally would, in your `.env` (e.g. `$OPENAI_API_KEY`,
  `$CARTESIA_API_KEY`, `$DEEPGRAM_API_KEY`, ...). A bot whose keys are missing
  fails its eval.

Install the framework with the eval extras (Kokoro, Moonshine, Whisper,
Ollama, and the services the bots use):

```sh
uv sync --group dev --all-extras --no-extra gstreamer --no-extra local
```

## Running

```sh
./run.sh                  # everything in the manifest
./run.sh -p voice-openai  # only bots whose path contains "voice-openai"
./run.sh -s capital_question  # only the capital_question scenario
./run.sh -c 8             # 8 at a time
./run.sh -n nightly       # output to test-runs/nightly/ instead of a timestamp
```

`run.sh` is a thin wrapper over `pipecat eval suite`; it always passes `-d` so
the full per-pipeline debug logs are saved (see below), and forwards any extra
flags:

```sh
uv run python -m pipecat.evals suite -d manifest.yaml [-p PATTERN] [-s SCENARIO] [-c N] [-n NAME] [-t SECS] [-a] [--no-cache]
```

Each run writes to `test-runs/<name>/` (a timestamp when `-n` is omitted):

- `logs/<bot>__<scenario>.log` — the bot subprocess output.
- `logs/<bot>__<scenario>.eval.log` — the harness's decision trace (always
  written; invaluable for diagnosing a flake).
- `logs/<bot>__<scenario>.debug.log` — the harness's full per-pipeline logs
  (user speech / bot speech transcription / judge / harness), one section per
  pipeline. Written whenever `-d/--debug` is passed, which `run.sh` always does.
- `recordings/<bot>__<scenario>.wav` — the conversation audio for audio-mode
  scenarios. The manifest sets `record: true`, so these are produced by default;
  pass `-a/--audio` to force recording on if a manifest has it off.

Useful flags: `-c/--concurrency`, `-t/--timeout` (default per-expectation
timeout in seconds, for expectations without their own `within_ms`), and
`--no-cache` (re-synthesize user audio every turn instead of reusing the cache).
Everything in the manifest header except the `suite:` list can also be overridden
on the command line (the command line wins) — `--bots-dir`, `--scenarios-dir`,
`--runs-dir`, `--base-port`, `--cache-dir`, `--spawn`, `--python` — so a manifest
can be just a `suite:` list with the rest supplied as flags.

### Concurrency and GPU

Only the judge LLM runs on the GPU. Ollama keeps one copy of the judge model
resident (`gemma2:9b` is ~7.4GB), so GPU use is roughly constant (~8.5GB peak)
regardless of `-c/--concurrency`. The user's voice (Kokoro) and the bot-speech
transcriber (Moonshine by default) both run on the CPU via ONNX Runtime, so they
cost no GPU memory; concurrency is bounded by CPU and RAM rather than GPU. A
16GB GPU (e.g. an RTX A4000) runs the default setup comfortably; swapping in a
much larger judge is what would pressure GPU memory, and an out-of-memory run
surfaces as a harness error in that run's `.eval.log`.

Whisper is available as an alternative transcriber (`transcription: {service:
whisper}`); it also defaults to the CPU (`device: cpu`, see `whisper_service`),
and can be put on the GPU with `device: cuda` if you have headroom.

## Running one scenario against an already-running bot

If you already have a bot running with `-t eval`, run a scenario directly
(handy while iterating on a scenario or a single bot):

```sh
pipecat eval run scenarios/capital_question.yaml --bot-url ws://localhost:7860
```

## Scenarios

A scenario is a sequence of `turns`. A turn sends a `user` utterance, presses
DTMF keys with `dtmf:` (mutually exclusive with `user:`), or is
observation-only (neither field) and just asserts — used for bot-first turns
like an opening greeting. The full file
format (events, expectations, `send_after:`, `image:`, ...) is documented in the
[`pipecat.evals.scenario`](../../src/pipecat/evals/scenario.py) module docstring.

Two things worth knowing when authoring:

- **Modality.** `judge:` and `user:` blocks select audio vs text. In audio mode
  the user's turns are synthesized (exercising the bot's STT for real) and the
  judge evaluates a local transcription of the bot's actual audio; text mode
  sends/judges text directly and is faster and silent.
- **Greet first.** A bot that greets on connect (most do) needs that greeting to
  finish before the first user turn — otherwise the question barges into it. So
  user-first scenarios lead with a bot-first turn that expects the greeting.

Shared `judge:`/`user:` config lives in small fragment files
(`judge_audio.yaml`, `judge_text.yaml`, `user_audio.yaml`) that scenarios pull in
with `!include` (resolved relative to the scenario file):

```yaml
user: !include user_audio.yaml
judge: !include judge_audio.yaml
```

## Vision (image input)

Some bots need session data they'd normally get from a `/start` request body,
such as a vision bot's image. The eval transport has no such endpoint, so a
bot entry can point to a JSON `runner_body:` file (resolved relative to the
manifest) that is passed to the bot as `--runner-body`:

```yaml
- bot: vision/vision-openai.py
  runner_body: scenarios/vision-cat.json   # {"image_path": "../assets/cat.jpg", "question": "..."}
  scenarios: [vision_describe]
```

The bot is spawned with the body file's directory as its working directory, so
a relative `image_path` in the body resolves next to the file and the two travel
together. The `vision_describe` scenario is a bot-first turn (no user input): the
bot describes the image (a cat) on connect and the judge checks that it described
a cat.

For function-calling-video bots, a turn can instead register an `image:` that the
eval transport serves when the bot requests a user image mid-conversation (see
`describe_image`).

## Adding coverage

- New bot: add an entry to `manifest.yaml` (`bot:` + the `scenarios:` it should run).
- New behavior to test: add a `scenarios/<name>.yaml` and reference it from the manifest.
