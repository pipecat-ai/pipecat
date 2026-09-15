# Turn-completion marker evals

`filter_incomplete_user_turns` asks the LLM to begin every response with a
marker that says whether the user's turn was complete (`●`, then the reply),
cut off mid-thought (`◐`, alone) or in need of more time (`○`, alone). The
protocol lives in `pipecat.turns.user_turn_completion_mixin`, and it only works
as well as the model follows the instructions. This suite measures that across
every text LLM service in Pipecat, so a prompt change can be checked against
the whole matrix and a new model can be scored as soon as it appears.

It is separate from the release evals in `../release-evals/`: those drive whole
example bots over audio, one scenario per behavior; this one drives the LLM
service alone, hundreds of short conversations per model, and asks one narrow
question of each answer.

## How it works

`run.py` builds the real Pipecat LLM service for each model (so the adapter's
role conversion, the system-instruction composition and the mixin's marker
parsing are all exercised), sends it a scripted conversation with the marker
instructions enabled, and records both the raw text stream and what the mixin
made of it: the marker frame it emitted, the text it let through as speech, and
any tool calls. No audio, no transport, no aggregators; the conversation history
is written the way the assistant aggregator would have stored it (`● reply`,
bare `◐` / `○`).

Every case is a short conversation. A user turn with an `expect:` runs an
inference and is scored; the other turns are history. Cases in the
`STT_VARIANT_CATEGORIES` also run as an STT-style transcript (lowercase, no
punctuation), since that is what a bot's LLM actually sees and the punctuated
examples in the prompt are the easy version.

```sh
uv run python scripts/turn-completion-evals/run.py                     # everything
uv run python scripts/turn-completion-evals/run.py -m openai -m groq   # some providers
uv run python scripts/turn-completion-evals/run.py -c cutoff -c drift  # some cases
uv run python scripts/turn-completion-evals/run.py --repeat 3 --judge  # rates, judged replies
uv run python scripts/turn-completion-evals/run.py --prompt terse      # a prompt variant
uv run python scripts/turn-completion-evals/run.py --markers "✓…?"     # other marker characters
uv run python scripts/turn-completion-evals/run.py --list-models / --list-cases
```

A run writes to `runs/<name>/` (a timestamp unless `-n` is given):

- `<provider>__<model>.jsonl`, one record per scored step: input, expectation,
  the framework's verdict, the raw stream, the spoken text, tool calls, failure
  kinds, latencies.
- `summary.md`: a leaderboard, pass rate per category and per input variant,
  the hardest steps across models, and the provider errors.
- `config.json`: the instructions and markers the run used.

Keys come from the repo's `.env`. A provider whose variable is missing is
skipped and listed in the summary.

## What is scored

Each scored step gets a **verdict** (what the mixin would do: `complete`,
`short`, `long`, or `none`) and a set of **failure kinds**:

| kind | meaning |
| --- | --- |
| `false_complete` | expected ◐/○, the model said ● and the bot would talk over the user |
| `false_incomplete` | expected ●, the model said ◐/○ and the bot goes silent |
| `wrong_incomplete_type` | ◐ for ○ or the reverse; only the timeout differs, so it is soft and does not fail the step |
| `missing_tool_call` / `unexpected_tool_call` | a tool case's call did or did not happen |
| `no_marker` | no marker at all; the mixin pushes the text with a warning |
| `marker_not_first` | text (or a stray byte) before the marker; the mixin still finds it but loses the prefix |
| `multiple_markers` | more than one marker in the response |
| `bare_complete` | ● with no text after it, the dead-air case of pipecat-ai/pipecat#5151 |
| `text_after_incomplete` | ◐/○ followed by text; the mixin suppresses it, but it costs tokens and shows weak adherence |
| `judge_no` | with `--judge`, a ● reply failed the case's `eval:` criterion |
| `error` / `timeout` | the provider failed or took longer than `-t` |

A step **passes** when the verdict matches, the format is clean and the judge
(if on) agrees. The leaderboard shows the pass rate and, separately, the
verdict-only and format-only rates, since a prompt fix for one rarely fixes the
other. Latencies are measured from the context frame reaching the service to
the first chunk and to the chunk holding the marker; they include the
harness's own overhead and connection setup, so compare them within a run, not
against a bot.

## Cases

`cases/<category>.yaml` holds a list of cases:

```yaml
category: cutoff
cases:
  - name: cutoff_preposition
    tools: [get_current_weather]        # optional, from TOOLS in run.py
    system: "..."                        # optional base prompt override
    stt_variant: false                   # optional; default by category
    turns:
      - bot: "Where would you go?"       # history; stored as "● Where would you go?"
      - user: "I'd go to Japan because"  # history only (no expect)
      - bot: "◐"                          # history; stored bare
      - developer: $reprompt_short       # the mixin's timeout re-prompt
        expect: complete                 # scored: the developer turn runs an inference
      - user: "the culture and the food."
        expect: complete                 # complete | short | long | incomplete | tool:<name>
        eval: "engages with Japan"       # optional judge criterion for ● replies
```

The categories: `complete`, `short_complete` (one-word answers, the
false-incomplete trap), `cutoff`, `time_request`, `preamble` (grammatically
complete, conversationally empty), `continuation` (fragment, marker, then the
rest), `reprompt` (the developer re-prompt after a timeout), `drift` (long
histories of bare markers, the conditioning behind #5151), `tools`, `greeting`,
`language`, `long`, and `hard` (edge cases where models disagree).

`--mode canonical` (default) builds each step's history from the case's own
turns, so every step is the same question for every model. `--mode live` feeds
each model its own earlier answers forward, the way the aggregator would, which
shows drift as it actually happens.

## Models

`models.yaml` lists a provider per Pipecat LLM service with the models to run,
their constructor arguments, and any provider config (thinking off, reasoning
effort). Per-model flags: `tools: false` skips the tool cases for a model whose
endpoint rejects them; `developer_role: false` sends developer messages as
user for a model whose chat template rejects the role (the Qwen templates on
Groq and Cerebras do); `system_suffix` appends text to the system prompt (the
Nemotron `/no_think` switch).

## Prompt iteration

The default instructions are the framework's. `--prompt <name>` loads
`prompts/<name>.txt` instead, a template with `{complete}`, `{short}` and
`{long}` placeholders, and `--markers` swaps the characters. Run the matrix
with the current prompt first, then a variant, and compare the two runs:

```sh
uv run python scripts/turn-completion-evals/compare.py runs/baseline runs/prompt-v2
```

`compare.py` lines the runs up step by step (only steps scored in both count),
and prints the per-model delta, the change in each failure kind, the change per
category, and the steps that moved most. The goal is to lift the weakest models
without moving the strong ones.

`prompts/` keeps the variants that have been tried. `v4.txt` is the text the
framework ships as its default; `v2.txt` and `v3.txt` are the steps that led
there, kept because their failure modes are instructive: examples written as
arrows were copied as arrows, examples written as a transcript made weak models
invent the next `User:` line, and a list of English cutoff words made models
stop recognising cutoffs in other languages. Prose examples avoided all three.
