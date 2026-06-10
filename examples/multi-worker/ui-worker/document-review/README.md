# document-review

The synthesis demo. A voice-driven workspace where the user reviews a
draft article — combining the patterns from every prior demo into one
application: snapshot reading, deixis (read + write), form-fill
state-changing actions, async job-group fan-out with progress streaming,
plus one custom command and one client-emitted event.

## What it shows

- **Read-side deixis**: select a paragraph, ask "review this", and the
  worker grounds in the selected text.
- **Async fan-out**: a paragraph review spawns two peer workers (clarity
  + tone) in parallel via `start_ui_job_group`. The in-flight card
  streams each worker's progress.
- **Custom UI command**: as each worker completes, `on_job_response`
  emits an `add_note` command with the worker's feedback; the client
  renders a note attached to the reviewed paragraph.
- **State-changing actions**: dictating a note fills the textarea and
  clicks Save (`fills` + `click` from the bundled `reply` tool).
- **Write-side deixis**: "where does it talk about rhythms?" → the worker
  finds the paragraph and uses `select_text` to put the page selection
  on it.
- **Client-emitted UI event**: clicking a note sends a `note_click` event
  back; the worker's `@ui_event("note_click")` handler dispatches
  `select_text` to jump to the paragraph. The round-trip event/command
  pattern.
- **Two LLM tools coexisting**: `ReplyToolMixin`'s `reply` handles normal
  turns; a custom `start_review` tool handles review kick-off. The prompt
  steers the model to pick one (single tool call per turn).
- **`on_job_response` interception**: the worker overrides this hook to
  translate reviewer responses into `add_note` commands — the peers don't
  know they're driving a UI; the worker mediates.

## What's new vs. the prior demos

| Prior demo | Pattern |
|---|---|
| hello-snapshot | snapshot streaming, voice/UI delegation |
| deixis | scroll, highlight + bidirectional text selection |
| form-fill | fills + click |
| async-tasks | job-group fan-out + cancel |

This one stitches all four together, plus the two patterns no prior demo
touched: a **custom UI command** (`add_note`) and a **custom
client-emitted event** (`note_click`).

## Run

Two terminals.

**Terminal 1 — bot:**

```bash
cd examples/multi-worker/ui-worker/document-review
uv run bot.py
```

The bot starts on `http://localhost:7860`.

**Terminal 2 — client:**

```bash
cd examples/multi-worker/ui-worker/document-review/client
npm install            # one-time
npm run dev
```

Open `http://localhost:5173` and click **Connect**.

## What to try

The article is a 6-paragraph draft seeded with one too-dense paragraph,
one too-vague one, and one with absolutist tone problems.

**Review flow (the centerpiece):**

- Select the run-on paragraph, say _"review this."_ — the worker
  acknowledges, the in-flight card appears, both reviewers tick through
  progress, and two notes attach to the paragraph (clarity flags the
  density).
- Select the absolutist paragraph, say _"give me feedback."_ — tone
  flags the strong words.

**Notes flow:**

- _"Add a note that this paragraph is too jargony."_ (with a paragraph
  selected) — the worker fills the textarea and clicks Save.
- Click any note in the panel — the page scrolls and selects the
  paragraph it was attached to.

**Navigation:**

- _"Where does it talk about structured rhythms?"_ — the worker jumps to
  the paragraph by selecting it.

**Cancellation:**

- During a review, click Cancel on the in-flight card. The reviewers'
  responses come back as `cancelled`; feedback that already arrived stays
  as a note.

## Requirements

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY`

A `.env` in the example folder is the easiest way to set these (see
`examples/multi-worker/env.example`).

## What this example does _not_ show

Real worker integrations (the reviewers compute simple text metrics — for
real LLM reviewers, swap them for `LLMWorker` subclasses whose
`on_job_request` runs the LLM with the paragraph text and a critique
prompt; everything else stays the same), note persistence, or
multi-document / multi-page flows.
