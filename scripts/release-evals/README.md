# Release Evals

Before a Pipecat release we make sure all (or most) of the 100+ examples still
work. Doing that by hand is slow and painful, so these "release evals" drive each
example automatically.

## How it works

Each example is a Pipecat **bot**. We run it with its eval transport
(`-t eval`), and the **eval harness** (`pipecat.evals`) connects to it as an RTVI
client, plays the user's turns (synthesizing audio when a scenario is in audio
mode), transcribes the bot's speech, and judges the response with an LLM.

A scenario (`scenarios/<name>.yaml`) is a scripted conversation plus the expected
results. For example the `simple_math` scenario asks "What is two plus two?" and
judges that the reply says four. Scenarios are reusable, so one shared scenario
covers many bots.

[`manifest.yaml`](manifest.yaml) maps each bot to the scenarios it runs.

## Running

```sh
./run.sh                  # everything in the manifest
./run.sh -p voice-openai  # only bots whose path contains "voice-openai"
./run.sh -s simple_math   # only the simple_math scenario
./run.sh -c 8 -a          # 8 at a time, record audio
./run.sh -n nightly       # output to test-runs/nightly/ instead of a timestamp
./run.sh -d               # also save the harness's full per-pipeline debug logs
```

Everything except the `suite:` list can be set in the manifest *or* on the
command line (the command line wins): `--bots-dir`, `--scenarios-dir`,
`--runs-dir`, `-c/--concurrency`, `--base-port`, `--cache-dir`, `--spawn`,
`--python`, `-a`, and `-d/--debug`. So a manifest can be just a `suite:` list
with the rest supplied as flags.

`run.sh` is a thin wrapper over the `pipecat eval suite` command:

```sh
uv run python -m pipecat.evals suite manifest.yaml [-p PATTERN] [-s SCENARIO] [-c N] [-a] [-d]
```

Each run writes, under `test-runs/<timestamp>/logs/`:

- `<bot>__<scenario>.log` — the bot subprocess output.
- `<bot>__<scenario>.eval.log` — the harness's decision trace (always; invaluable
  for diagnosing a flake).
- `<bot>__<scenario>.debug.log` — with `-d/--debug` only: the harness's full
  per-pipeline logs (voice / transcription / judge / harness), in one file with a
  section per pipeline.

## Running one scenario against an already-running bot

If you already have a bot running with `-t eval`, run a scenario directly:

```sh
pipecat eval run scenarios/simple_math.yaml --bot-url ws://localhost:7860
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

## Adding coverage

- New bot: add an entry to `manifest.yaml` (`bot:` + the `scenarios:` it should run).
- New behavior to test: add a `scenarios/<name>.yaml` and reference it from the manifest.
