# deixis

The voice LLM asks the UI worker what the user selected, and points back.
Select a paragraph in the article and ask "explain this". The voice LLM
cannot see the page, so it asks the `UIWorker` for the selected text and
answers from it. Ask "where does it talk about RNA editing?" and the
worker selects that paragraph on the page, so you see what the bot means.

## What it shows

- **The read direction.** The client captures `window.getSelection()`
  and sends it with each snapshot. The voice LLM calls `selection()`,
  a tool that sends a `selection` job to the worker, and gets the
  selected text back as short data. It never sees the page.
- **The write direction.** For "where does it talk about X" the voice
  LLM calls `screen("select_text", "the paragraph about X")`. The
  worker's classifier picks the paragraph the words mean, one choice
  question over the elements on screen, and sends the `select_text`
  command. The client selects the paragraph and scrolls to it.
- **A UIWorker with a classifier and no LLM turn.** `DeixisWorker`
  answers the `selection` job from the snapshot with plain code and the
  built-in `screen` job with its classifier. With `TYPESAFE_API_KEY` set
  the classifier is Jev; otherwise the worker's own LLM answers through
  an `LLMClassifier`.

## Architecture

```
Main worker (PipelineWorker, owns transport + RTVI):
  transport.in → STT → user_agg → LLM → TTS → transport.out → assistant_agg
    ├── selection() tool          → job "selection" on the UI worker
    └── screen(action, target)    → job "screen" on the UI worker

DeixisWorker (UIWorker with a classifier, no LLM turn):
  ├── @job("selection"): the selected text, read from the snapshot
  └── built-in "screen" job: select_text / scroll_to / highlight by description
```

## Run

Two terminals.

**Terminal 1: bot**

```bash
cd examples/multi-worker/ui-worker/deixis
uv run bot.py
```

The bot starts on `http://localhost:7860`.

**Terminal 2: client**

```bash
cd examples/multi-worker/ui-worker/deixis/client
npm install            # one-time
npm run dev
```

Open `http://localhost:5173` and click **Connect**.

## What to try

The page renders a short essay on octopus cognition with selectable
paragraphs.

**Read direction (you select, the bot answers about it):**

- Select the paragraph about RNA editing, then _"What does this mean?"_
- Select any paragraph, then _"Explain this in one sentence."_
- With nothing selected, _"Explain this."_ The bot asks you to select
  something, because the tool told it nothing is selected.

**Write direction (the bot points back):**

- _"Where does it talk about how octopuses solve problems?"_ The bot
  says where it is and the page selects that paragraph.
- _"Show me the part about the skin."_ Same, by description.

## Requirements

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY`
- `TYPESAFE_API_KEY` (optional; uses Jev as the classifier)

A `.env` in the example folder is the easiest way to set these (see
`examples/multi-worker/env.example`).

## What this example _doesn't_ show

Form filling (see `form-fill/`), async task cards (see `async-tasks/`),
or custom command handlers beyond `scroll_to` / `highlight` /
`select_text`.
