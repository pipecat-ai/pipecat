# shopping-list

Every voice turn drives the UI; speech is incidental. The user builds a
shopping list by talking — "add milk and eggs", "check off the bread",
"drop the last one", "what's left?" — and the list updates on screen
every turn. The assistant may also say something back. Updates run in
parallel on separate workers; the voice layer never mutates the list.

This is the pattern for "every input acts, may speak."

## What it shows

- **A standard voice pipeline + a UIWorker.** The voice layer is an
  ordinary `transport → STT → LLM → TTS` pipeline whose LLM converses and
  never *mutates* the list. Its user aggregator fires
  `on_user_turn_stopped` once per user turn, and that handler dispatches
  the transcript to the UIWorker as a `respond` job
  (`worker.job("ui", name="respond", payload={"query": transcript})`) —
  a bus message. The UIWorker does all the list work, silently, on its
  own worker, so its LLM output never reaches TTS.
- **Snapshot-driven action.** Before each UIWorker inference, the current
  `<ui_state>` is auto-injected (via the LLM's `on_before_process_frame`
  hook), so the worker resolves "the milk", "the last one", "the checked
  ones" against the live list.
- **Custom UI commands.** A single bundled `update_list` tool maps the
  request to `add_item` / `set_checked` / `remove_item` commands (plus the
  standard `highlight`, used to *show* what's left). Each item is a
  checkbox whose accessible name is the item text, so the snapshot exposes
  every item's label and checked state.
- **The voice layer reads the list on demand.** It has one read-only tool,
  `check_list`, that reads the UIWorker's live snapshot
  (`ListWorker.list_summary`) and answers "what's left?" / "what's on my
  list?" from what's *really* on screen — including items the user checked
  off or added by hand. The snapshot is the shared source of truth: the
  UIWorker acts on it, the voice layer reads it.

## What it adds vs. the prior demos

The other demos use the **request/response delegation** shape: the voice
LLM decides, via a tool call, when to consult the UIWorker. Here every user
turn *mutates* the list automatically (the `on_user_turn_stopped` handler,
not a tool), and the voice LLM runs independently for conversation. It
keeps just one small tool — `check_list` — for *reading* the list when the
user asks, since it otherwise can't see the screen.

## Run

Two terminals.

**Terminal 1 — bot:**

```bash
cd examples/multi-worker/ui-worker/shopping-list
uv run bot.py
```

The bot starts on `http://localhost:7860`.

**Terminal 2 — client:**

```bash
cd examples/multi-worker/ui-worker/shopping-list/client
npm install            # one-time
npm run dev
```

Open `http://localhost:5173` and click **Connect**.

## What to try

- _"Add milk and a dozen eggs."_ — both items appear; the voice
  acknowledges.
- _"Add bread, butter, and coffee."_ — three more land.
- _"Check off the bread."_ — it gets ticked and struck through.
- _"Actually, drop the butter."_ — removed.
- _"Clear the ones I've already got."_ — removes everything checked.
- _"What's left?"_ — the unchecked items pulse, and the voice reads them
  out (via `check_list`).
- _"Hi there!"_ — the voice greets you; the list is unchanged (the
  UIWorker no-ops).

You can also **check a box or type a new item by hand**, then ask _"what's
on my list?"_ — the voice answers from the real list, including your manual
edits, because `check_list` reads the live snapshot.

## Requirements

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY`

A `.env` in the example folder is the easiest way to set these (see
`examples/multi-worker/env.example`).

## What this example _doesn't_ show

The voice layer sees the list only through the narrow `check_list` tool —
a flat item/checked summary, not the full page (reading the whole screen is
the UIWorker's job). State flows one way for changes (voice → UIWorker,
never the reverse), and the list isn't persisted — refresh and it's gone.
