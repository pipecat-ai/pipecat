# form-fill

A voice-guided, **accessibility-first** form walkthrough. Instead of
waiting for the user to dictate values, the assistant *leads*: it walks
the user through a job application one section at a time — personal
information (name, email, phone), then job qualifications (years of
experience and why they're interested), then submit — confirming what it
captured before moving on. A user who can't see the screen never has to.

## What it shows

- **A proactive, guided flow.** The voice LLM greets on connect and asks
  for the user's name; from there it writes each answer into the form,
  confirms what it heard, and asks for the next piece, section by section.
  All the guidance lives in one prompt, `VOICE_PROMPT`.
- **One screen tool.** The voice LLM has a single tool, `screen(action,
  target, value)`, from `screen_tool("ui")`. `screen("list", "textbox")`
  returns the inputs with their current values, `screen("fill", "the
  email field", "john@example.com")` writes a value, and `screen("click",
  "the submit button")` submits. It never sees the page.
- **A UIWorker with a classifier and no LLM turn.** `FormWorker` answers
  the `screen` job: for "the email field" it asks its classifier which
  element on the live `<ui_state>` the words mean, then sends the command.
  With `TYPESAFE_API_KEY` set the classifier is Jev; otherwise the
  worker's own LLM answers through an `LLMClassifier`.
- **Driven by the form, not hidden state.** Each turn the voice LLM lists
  the inputs and steers toward the next empty one. Progress is the form;
  there is no separate step counter to keep in sync.

## What it adds vs. `deixis`

`deixis` is reactive: it answers one-off questions, and its UI worker's
LLM reads the page to do so. This one is proactive: the voice LLM drives
a multi-step flow and changes the form through the screen tool, with the
UI worker only grounding and acting.

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
