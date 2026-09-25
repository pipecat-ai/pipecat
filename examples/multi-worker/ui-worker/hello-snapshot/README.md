# hello-snapshot

The smallest `UIWorker` example. A static HTML page with a few news cards
and a sidebar. The voice LLM cannot see the page; it asks the `UIWorker`
about it, and speaks the answer.

## What it shows

- **The accessibility snapshot.** The client walks the DOM and streams
  a snapshot of the page. `PipelineWorker` passes it to the `UIWorker`
  on its own (RTVI is enabled by default), and the worker keeps the
  latest one.
- **Asking the UI worker a question.** The voice LLM has one tool,
  `ask_page(question)`, which sends the worker's built-in `respond` job.
  The worker's LLM sees the latest snapshot, answers in a sentence or
  two through its `answer` tool, and the answer comes back to the voice
  LLM as the tool's result. The voice LLM never sees the page.

## Architecture

```
Main worker (PipelineWorker, owns transport + RTVI):
  transport.in → STT → user_agg → LLM → TTS → transport.out → assistant_agg
    └── ask_page(question) tool → job "respond" on the UI worker

HelloWorker (UIWorker):
  └── @tool answer(text) → the job's response
```

## Run

Two terminals.

**Terminal 1: bot**

```bash
cd examples/multi-worker/ui-worker/hello-snapshot
uv run bot.py
```

The bot starts on `http://localhost:7860`.

**Terminal 2: client**

```bash
cd examples/multi-worker/ui-worker/hello-snapshot/client
npm install            # one-time
npm run dev
```

Open `http://localhost:5173` and click **Connect**.

## What to try

Once connected, ask:

- _"What's on this page?"_ A summary of the layout (heading, three
  stories, trending tags sidebar).
- _"What was the second story about?"_ The snapshot keeps reading
  order, so "second" resolves cleanly.
- _"Which story was about energy?"_ The worker answers from the
  stories' content, not just their titles.
- _"What tags are trending?"_ Reads the sidebar.
- _"What's the capital of France?"_ The worker answers from general
  knowledge when the question has nothing to do with the page.

If you scroll the page (in a smaller window) or resize, the snapshot is
sent again. Elements that are off screen are marked as such, so
"what do I see right now" answers about the visible part.

## Requirements

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY`

A `.env` in the example folder is the easiest way to set these (see
`examples/multi-worker/env.example`).

## What this example _doesn't_ show

Acting on the page (`scroll_to`, `highlight`, ...), the classifier-backed
`screen` tool, form filling, selection-based deixis, or async task
cards. The other examples in this folder build on this same skeleton.
