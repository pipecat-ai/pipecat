# deixis

The UIWorker grounds in what the user just selected. Highlight a
paragraph in the article and ask "explain this" — the worker reads your
selection from the snapshot and answers about that specific content.

## What it shows

- The **read direction**: the client captures `window.getSelection()`
  and emits a `<selection ref="...">selected text</selection>` block
  inside `<ui_state>`. The `UIWorker` treats it as the deictic referent
  for "this", "that", "this paragraph". Asking "what does this mean?"
  with a paragraph selected resolves cleanly.
- The **write direction**: the worker says "this paragraph" and issues a
  `select_text=ref` command. The client puts the page's text selection
  on that element, so the user sees exactly which paragraph the worker
  means.
- `ReplyToolMixin`'s pointing/reading fields: the bundled `reply` tool
  offers `scroll_to`, `highlight` (brief flash), and `select_text`
  (durable selection), used per turn as the request needs.

## What it adds vs. `hello-snapshot`

`hello-snapshot` proved the worker can *read* the page and answer. This
one proves it can *act* on it: scroll and highlight to point, and
(uniquely) read the user's text selection and point back via a
programmatic selection. The new parts are the `scroll_to` / `highlight` /
`select_text` commands and their client handlers.

## Run

Two terminals.

**Terminal 1 — bot:**

```bash
cd examples/multi-worker/ui-worker/deixis
uv run bot.py
```

The bot starts on `http://localhost:7860`.

**Terminal 2 — client:**

```bash
cd examples/multi-worker/ui-worker/deixis/client
npm install            # one-time
npm run dev
```

Open `http://localhost:5173` and click **Connect**.

## What to try

The page renders a short essay on octopus cognition with selectable
paragraphs.

**Read direction (user selects, worker grounds):**

- Select the paragraph about RNA editing → _"What does this mean?"_
- Select any paragraph → _"Explain this in one sentence."_

**Write direction (worker points back):**

- _"Where does it talk about how octopuses solve problems?"_ (no
  selection) — the worker finds the paragraph, speaks a brief reply, and
  selects it for you.
- _"How many neurons does an octopus have?"_ — answers and selects the
  source paragraph.

**Conversational without pointing:**

- _"What's this article about?"_ — a one-sentence summary, no selection.

## Requirements

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY`

A `.env` in the example folder is the easiest way to set these (see
`examples/multi-worker/env.example`).

## What this example _doesn't_ show

Form filling (see `form-fill/`), async task cards (see `async-tasks/`),
or custom command handlers beyond `scroll_to` / `highlight` /
`select_text`.
