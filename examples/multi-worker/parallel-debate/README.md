# parallel-debate

Parallel fan-out using `worker.job_group(...)`. A voice bot takes a topic from the user, kicks off three workers in parallel (advocate, critic, analyst), waits for all three to respond, and synthesizes a balanced answer. Each worker keeps its own LLM context across rounds.

See the [top-level multi-worker README](../README.md) for setup and shared environment variables.

## Running

```bash
uv run parallel-debate/parallel-debate.py
```

Open <http://localhost:7860/client> in your browser to talk to your bot.

To use Daily transport:

```bash
uv run parallel-debate/parallel-debate.py --transport daily
```

## Architecture

```
Main worker (transport + LLM + `debate` tool)
  └── job_group(advocate, critic, analyst)
        └── DebateWorker (LLMContextWorker, one per role)
```

- **Main worker**: transport (STT, TTS) + LLM moderator with a `debate` direct function that fans out via `worker.job_group(...)`.
- **Debate workers**: `LLMContextWorker`s spawned on the runner. Each keeps its own `LLMContext` across rounds and ships its completed turn back as a job response via the assistant-aggregator's `on_assistant_turn_stopped` event.
