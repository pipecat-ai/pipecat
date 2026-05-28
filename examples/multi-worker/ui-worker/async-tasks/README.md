# async-tasks

The UIWorker fans out long-running work to multiple peer workers in
parallel, streams their progress to an in-flight panel on the page, and
lets the user cancel mid-flight.

## What it shows

- The **`ui_job_group` / `start_ui_job_group`** API: dispatching
  parallel work to multiple peer workers and automatically forwarding
  every job lifecycle event to the client. The `reply` tool calls
  `start_ui_job_group("wikipedia", "news", "scholar", payload=...,
  label=...)` and the `UIWorker` does the rest.
- The four **`ui-job-group` envelopes** the worker forwards (`group_started`,
  `job_update`, `job_completed`, `group_completed`) and the
  client-side `RTVIEvent.UIJobGroup` event for consuming them. The client
  keeps a state map keyed by `job_id` and renders per-worker progress.
- **Cancellation**: the in-flight card's Cancel button calls
  `client.cancelUIJobGroup(job_id, reason)`. The reserved `__cancel_job_group`
  event is translated by the `UIWorker` into `cancel_job_group(job_id)`
  on the registered group; cancelled workers report status `cancelled`.
- **Background dispatch from a tool**: `start_ui_job_group` returns
  immediately so the `reply` tool can speak its acknowledgement
  ("Researching the Mariana Trench now") while the workers run — the
  main LLM is free to take follow-up turns.

## What it adds vs. the prior demos

The other examples use the request/response half of the bus protocol
(main LLM → UIWorker → reply). This one adds the streaming job-group
half: UIWorker → peer workers → progress events forwarded to the client.
The architecture grows from "one delegate" to "one delegate plus a
worker pool" — the peers are plain `BaseWorker`s launched on the runner.

## Run

Two terminals.

**Terminal 1 — bot:**

```bash
cd examples/multi-worker/ui-worker/async-tasks
uv run bot.py
```

The bot starts on `http://localhost:7860`.

**Terminal 2 — client:**

```bash
cd examples/multi-worker/ui-worker/async-tasks/client
npm install            # one-time
npm run dev
```

Open `http://localhost:5173` and click **Connect**.

## What to try

The workers are simulated (canned summaries, randomized `asyncio.sleep`
delays) so the demo focuses on the protocol, not the AI. Each research
call takes a few seconds.

- _"Research the Mariana Trench."_ — the worker spawns three peers,
  acknowledges in one short reply, and a card appears showing each
  peer's status as it progresses (searching → found N results →
  summarizing → completed).
- _"Look up octopus cognition."_ — same flow; a second card stacks.
- _"Research the moon, then research Mars."_ — two groups run
  concurrently.
- _"How are you?"_ (no research) — quick reply, no job group.
- **Click Cancel on an in-flight card** — the cancellation routes
  through, the peers' tasks raise `CancelledError`, and their responses
  come back as `cancelled`.

## Requirements

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY`

A `.env` in the example folder is the easiest way to set these (see
`examples/multi-worker/env.example`).

## What this example _doesn't_ show

Real worker integrations (the peers are simulated), LLM-driven peers
(these are pure data-fetch — a peer can itself be an `LLMWorker`),
streaming chunks (`send_job_stream_data` for progressive output), or
worker-to-worker fan-out (nested job groups).
