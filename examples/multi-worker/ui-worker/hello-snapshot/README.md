# hello-snapshot

The smallest possible `UIWorker` example. A static HTML page with a few
news cards and a sidebar. The user speaks; the worker answers grounded
in whatever's currently on screen.

## What it shows

- The accessibility-snapshot pipeline: the client walks the DOM and
  streams a snapshot, which the `UIWorker` injects into its LLM context
  as `<ui_state>`.
- The UIWorker delegate setup: the main pipeline's LLM (the
  conversational layer) delegates every utterance to a `HelloWorker`
  (`UIWorker`) via the `answer_about_screen` tool
  (`params.pipeline_worker.job("hello", name="respond", ...)`) and speaks
  the result.
- The native RTVI⇄bus UI wiring built into `PipelineWorker`: with
  `enable_rtvi=True` (the default), inbound `ui-snapshot` messages are
  broadcast on the bus and the `UIWorker` stores them — no decorator or
  manual wiring.

## Architecture

```
Main worker (PipelineWorker, owns transport + RTVI):
  transport.in → STT → user_agg → LLM → TTS → transport.out → assistant_agg
    └── answer_about_screen(query) tool
          └── params.pipeline_worker.job("hello", name="respond", payload={query})

HelloWorker (UIWorker):
  └── @tool answer(text)
```

## Run

Two terminals.

**Terminal 1 — bot:**

```bash
cd examples/multi-worker/ui-worker/hello-snapshot
uv run bot.py
```

The bot starts on `http://localhost:7860`.

**Terminal 2 — client:**

```bash
cd examples/multi-worker/ui-worker/hello-snapshot/client
npm install            # one-time
npm run dev
```

Open `http://localhost:5173` and click **Connect**.

## What to try

Once connected, ask the worker:

- _"What's on this page?"_ — it summarizes the layout (heading, three
  stories, trending tags sidebar).
- _"What was the second story about?"_ — sibling order in the snapshot
  matches reading order, so "second" resolves cleanly.
- _"Which story was about energy?"_ — the worker grounds against the
  actual content, not just titles.
- _"What tags are trending?"_ — exercises sidebar reading.
- _"What's the capital of France?"_ — the worker answers from general
  knowledge when the question has nothing to do with the page.

If you scroll the page (in a smaller window) or resize, the snapshot
re-emits. Off-screen elements get an `[offscreen]` tag the worker
respects when answering positional questions like "what do I see right
now."

## Requirements

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY`

A `.env` in the example folder is the easiest way to set these (see
`examples/multi-worker/env.example`).

## What this example _doesn't_ show

The read-side foundation only — no acting on the page (`scroll_to`,
`highlight`, ...), form filling, selection-based deixis, or async task
cards. Those build on this same skeleton.
