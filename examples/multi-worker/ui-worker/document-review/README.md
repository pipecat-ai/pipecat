# document-review

The synthesis demo. A voice-driven workspace where the user reviews a
draft article — combining the patterns from every prior demo into one
application: snapshot reading, deixis (read + write), form-fill
state-changing actions, async job-group fan-out with progress streaming,
plus one custom command and one client-emitted event.

## What it shows

- **The voice LLM leads, the UI worker grounds and acts.** The voice LLM
  has three tools: `review_selection()`, `add_note(text)` and the generic
  `screen(action, target, value)` from `screen_tools("ui")`. Each sends a
  job to `ReviewWorker` and returns short data. The voice LLM never sees
  the page, and the UI worker never runs an LLM turn.
- **Read-side deixis**: select a paragraph and ask "review this" or
  "explain this". The worker reads its `selection` from its own snapshot,
  and `screen("selection")` hands the text to the voice, so no tool needs
  a ref.
- **Async fan-out**: `review_selection` says "Reviewing this paragraph"
  through TTS, then runs two peer workers (clarity + tone) in parallel as
  a job group. The in-flight card streams each worker's progress, and
  when both have answered the voice gives their feedback.
- **Custom UI command**: as each reviewer completes, `on_job_response`
  emits an `add_note` command with its feedback; the client renders a
  note attached to the reviewed paragraph.
- **Grounded actions**: `add_note(text)` has the worker find the notes
  textarea and the Save button with its classifier, fill and click. The
  classifier is the worker's own LLM through an `LLMClassifier`; pass a
  `JevClassifier` for faster, calibrated answers.
- **Write-side deixis**: "where does it talk about rhythms?" is
  `screen("select_text", "the paragraph about rhythms")`; the classifier
  picks the paragraph by its text and the page selection lands on it.
- **Client-emitted UI event**: clicking a note sends a `note_click` event
  back; the worker's `@ui_event("note_click")` handler dispatches
  `select_text` to jump to the paragraph. The round-trip event/command
  pattern.

## What's new vs. the prior demos

| Prior demo | Pattern |
|---|---|
| hello-snapshot | snapshot streaming, voice/UI delegation |
| deixis | scroll, highlight + bidirectional text selection |
| form-fill | grounded fill + click through the screen tool |
| async-tasks | job-group fan-out + cancel, results back to the voice |

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
