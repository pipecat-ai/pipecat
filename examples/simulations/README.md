# Pipecat Simulations

A simulation puts an autonomous caller on the line with your bot. You describe
who the caller is and what they want; a persona LLM holds the conversation, ends
the call with its `end_call` tool once the goal is met or hopeless, and a judge
decides from the whole conversation whether the goal was achieved and how the
bot scored on the quality criteria you name. It answers "can my bot get this
caller what they need?", the question a scripted scenario (which only replays
fixed turns) cannot.

A simulation is the second kind of scenario file, told apart from a scripted one
by its `persona:`, and runs with the same `pipecat eval run` against a bot
started with its eval transport. Any of the example bots will do.

## Setup

From the repo root:

```bash
uv sync --all-extras
source .venv/bin/activate
```

The harness runs its judge and, in audio mode, the caller's voice and the bot's
transcription locally:

- **The judge**: [Ollama](https://ollama.com) with `gemma4:12b` pulled
  (`ollama pull gemma4:12b`); see `judge_text.yaml` and `judge_audio.yaml`.
- **The persona LLM**: `simulator.yaml` names OpenAI's `gpt-4o-mini`, so set
  `OPENAI_API_KEY`. The model must support function calling.
- **Audio mode**: Kokoro speaks the caller's turns and Moonshine transcribes the
  bot's; both download their models on first use (see `user_audio.yaml`).

The bot needs its own service keys in `.env`, as usual.

## Running

Start a bot with its eval transport, then run a simulation against it:

```bash
uv run python examples/voice/voice-cartesia.py -t eval --port 7860
uv run pipecat eval run examples/simulations/capital_curious.yaml --bot-url ws://localhost:7860 -v
```

The run prints the verdict with the quality score and how the call ended; with
`-v` it adds the judge's reason, each metric's reason, the caller's own view of
how it went, and the conversation. Add `-d` to keep the harness's full logs and
`-a` to record the audio of an audio-mode run.

## Files

- `capital_curious.yaml`: a text-mode caller who wants one fact and hangs up.
  The smallest simulation that exercises the whole loop.
- `capital_curious_audio.yaml`: the same caller, speaking and listening.
- `simulator.yaml`, `judge_text.yaml`, `judge_audio.yaml`, `user_audio.yaml`:
  the persona LLM, judge, and caller-voice blocks the simulations `!include`.

The full file format is documented in the `pipecat.evals.simulation` module.
