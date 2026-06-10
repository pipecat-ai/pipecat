# form-fill

A voice-guided, **accessibility-first** form walkthrough. Instead of
waiting for the user to dictate values, the assistant *leads*: it walks
the user through a job application one section at a time — personal
information (name, email, phone), then job qualifications (years of
experience and why they're interested), then submit — confirming what it
captured before moving on. A user who can't see the screen never has to.

## What it shows

- **A proactive, guided flow.** The voice agent greets on connect and
  asks for the user's name; from there it hands each answer to the UI
  worker, which writes it into the form with `fills`, reads back what it
  heard, and asks for the next piece — section by section.
- **Driven by `<ui_state>`, not hidden state.** `FormWorker` runs
  stateless (`keep_history=False`): each turn it sees which fields are
  already filled and steers toward the next empty one. Progress *is* the
  form — there's no separate step counter to keep in sync.
- **The worker owns every spoken line.** `FormWorker` composes
  `ReplyToolMixin`, which replies with verbatim TTS (`tts_speak=True`),
  so the worker's `answer` is spoken directly by the main pipeline's TTS
  and the voice LLM never paraphrases it. All the guidance lives in one
  prompt (`UI_PROMPT`).
- **The state-changing actions:** `fills` (a list of `{"ref","value"}`,
  so several fields can be written in one turn — "I'm John Smith" fills
  first AND last name) and `click` (to press submit at the end). Same
  `ReplyToolMixin` bundle that `deixis` uses, just exercising the
  input-writing fields.

## What it adds vs. `deixis`

`deixis` is *reactive* — it answers one-off questions and exercises the
visual fields of `reply` (`scroll_to`, `highlight`, `select_text`). This
one is *proactive*: the assistant drives a multi-step flow and exercises
the state-changing fields (`fills`, `click`). Same composition, same
mixin — different fields, and a leading
rather than reacting posture.

## Run

Two terminals.

**Terminal 1 — bot:**

```bash
cd examples/multi-worker/ui-worker/form-fill
uv run bot.py
```

The bot starts on `http://localhost:7860`.

**Terminal 2 — client:**

```bash
cd examples/multi-worker/ui-worker/form-fill/client
npm install            # one-time
npm run dev
```

Open `http://localhost:5173` and click **Connect**.

## What to try

Just click **Connect** and answer the assistant — it leads. A typical
session:

- _"I'm John Smith."_ — fills first and last name, then asks for your
  email.
- _"john at example dot com."_ — converts to `john@example.com`, then
  asks for your phone number.
- _"555 123 4567."_ — fills it, confirms your details are complete, and
  moves on to qualifications.
- _"Five years, and I love building real-time voice agents."_ — fills
  both qualification fields and asks if you're ready to submit.
- _"Yes."_ — clicks submit.

You can correct anything as you go (_"actually, my email is …"_) and the
assistant re-fills just that field.

## Requirements

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY`

A `.env` in the example folder is the easiest way to set these (see
`examples/multi-worker/env.example`).

## What this example _doesn't_ show

Selection-based deixis (see `deixis/`) or async task cards (see
`async-tasks/`).
