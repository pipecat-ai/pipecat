# shopping-list

The voice LLM asks, the UI worker finds and acts. The user builds a
shopping list by talking, "add milk and eggs", "check off the bread",
"drop the last one", "what's left?", and the list updates on screen. The
voice LLM understands the words; the UI worker grounds them on the page
with a classifier. No LLM turn runs on the UI side.

## What it shows

- **A standard voice pipeline with tools.** The voice layer is an
  ordinary `transport → STT → LLM → TTS` pipeline. Its LLM converses and
  calls `add_items`, `check_items`, `uncheck_items`, `remove_items`,
  `clear_checked` and `check_list`, each with the items in the user's
  words. Every tool sends a job to the UI worker and returns its answer
  as short data. The voice LLM never sees the screen.
- **A UIWorker with a classifier.** `ListWorker` answers the jobs. For
  an item named in words it asks its classifier which checkbox on the
  live `<ui_state>` the words mean, one choice question over the list,
  then sends `set_checked` or `remove_item`. `add_item` needs no
  classifier, since the voice LLM already carries the text. `summary`
  reads the snapshot with plain code. With `TYPESAFE_API_KEY` set the
  classifier is Jev, about a tenth of a second per question with a
  calibrated probability; otherwise the worker's own LLM answers through
  an `LLMClassifier`.
- **Custom UI commands.** `add_item`, `set_checked` and `remove_item` are
  the client's commands. Each item is a checkbox whose accessible name is
  the item text, so the snapshot exposes every item's label and checked
  state.
- **The snapshot is the source of truth.** `check_list` reads what is
  really on screen, including items the user checked off or added by
  hand, so the voice answers "what's left?" from the page, not from
  memory.

## What it adds vs. the prior demos

The other demos forward the user's words to a UIWorker whose LLM reads
the page and decides what to do. Here the voice LLM decides, and the UI
worker only grounds and acts: a classifier call of about 100 ms instead
of an LLM turn, and no prompt teaching a second LLM how to read refs.

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
- _"Check off the brown one."_ with only "bread" on the list — the
  classifier is not sure enough, the tool answers not found, and the
  voice asks which one you mean.

You can also **check a box or type a new item by hand**, then ask _"what's
on my list?"_ — the voice answers from the real list, including your manual
edits, because `check_list` reads the live snapshot.

## Requirements

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY`
- `TYPESAFE_API_KEY` (optional, for Jev)

A `.env` in the example folder is the easiest way to set these (see
`examples/multi-worker/env.example`).

## What this example _doesn't_ show

The voice layer sees the list only through the tools' short answers, never
the page. State flows one way for changes (voice tools → UI worker), and
the list isn't persisted — refresh and it's gone.
