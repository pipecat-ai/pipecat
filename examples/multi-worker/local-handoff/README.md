# local-handoff

Two LLM workers (greeter + support) that transfer control to each other during a voice conversation. A main worker owns the transport pipeline and bridges frames to the bus.

See the [top-level multi-worker README](../README.md) for setup and shared environment variables.

## Running

```bash
uv run local-handoff/local-handoff-two-agents.py
```

Open <http://localhost:7860/client> in your browser to talk to your bot.

To use Daily transport:

```bash
uv run local-handoff/local-handoff-two-agents.py --transport daily
```

## Overview

- **[`local-handoff-two-agents.py`](local-handoff-two-agents.py)** — Two LLM workers (greeter + support) that hand off via `activate_worker(..., deactivate_self=True)`. The main worker owns STT, TTS, transport, and a `BusBridgeProcessor`.
- **[`local-handoff-two-agents-tts.py`](local-handoff-two-agents-tts.py)** — Same shape, but each child worker ships with its own `CartesiaTTSService` in a custom pipeline. The main worker has no TTS — audio comes from whichever child is active over the bus.
