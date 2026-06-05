# Release Evals

Before a Pipecat release we make sure all (or most) of the 100+ examples still
work. Doing that by hand is slow and painful, so these "release evals" drive each
example automatically.

## How it works

Each example is a Pipecat **agent**. We run it with its eval transport
(`-t eval`), and the **eval harness** (`pipecat.evals`) connects to it as an RTVI
client, plays the user's turns (synthesizing audio when a scenario is in audio
mode), transcribes the agent's speech, and judges the response with an LLM.

A scenario (`scenarios/<name>.yaml`) is a scripted conversation plus the expected
results. For example the `simple_math` scenario asks "What is two plus two?" and
judges that the reply says four. Scenarios are reusable, so one shared scenario
covers many agents.

[`manifest.yaml`](manifest.yaml) maps each agent to the scenarios it runs.

## Running

```sh
./run.sh                  # everything in the manifest
./run.sh -p voice-openai  # only agents whose path contains "voice-openai"
./run.sh -s simple_math   # only the simple_math scenario
./run.sh -c 8 -a          # 8 at a time, record audio
./run.sh -n nightly       # output to test-runs/nightly/ instead of a timestamp
```

Everything except the `suite:` list can be set in the manifest *or* on the
command line (the command line wins): `--agents-dir`, `--scenarios-dir`,
`--runs-dir`, `-c/--concurrency`, `--base-port`, `--cache-dir`, `--spawn`,
`--python`, and `-a`. So a manifest can be just a `suite:` list with the rest
supplied as flags.

`run.sh` is a thin wrapper over the `pipecat eval suite` command:

```sh
uv run python -m pipecat.evals suite manifest.yaml [-p PATTERN] [-s SCENARIO] [-c N] [-a]
```

Per-agent logs and the harness's own decision trace (`<agent>__<scenario>.eval.log`,
invaluable for diagnosing a flake) land under `test-runs/<timestamp>/logs/`.

## Running one scenario against an already-running agent

If you already have an agent running with `-t eval`, run a scenario directly:

```sh
pipecat eval run scenarios/simple_math.yaml --agent-url ws://localhost:7860
```

## Vision (image input)

Some agents need session data they'd normally get from a `/start` request body,
such as a vision bot's image. The eval transport has no such endpoint, so an
agent entry can point to a JSON `runner_body:` file (resolved relative to the
manifest) that is passed to the agent as `--runner-body`:

```yaml
- agent: vision/vision-openai.py
  runner_body: scenarios/vision-cat.json   # {"image_path": "../assets/cat.jpg", "question": "..."}
  scenarios: [vision_describe]
```

The agent is spawned with the body file's directory as its working directory, so
a relative `image_path` in the body resolves next to the file and the two travel
together. The `vision_describe` scenario is a bot-first turn (no user input): the
bot describes the image (a cat) on connect and the judge checks that it described
a cat.

## Adding coverage

- New agent: add an entry to `manifest.yaml` (`agent:` + the `scenarios:` it should run).
- New behavior to test: add a `scenarios/<name>.yaml` and reference it from the manifest.
