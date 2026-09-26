# Turn-completion marker evals

`filter_incomplete_user_turns` asks the LLM to begin every response with a
marker that says whether the user's turn was complete (`●`, then the reply),
cut off mid-thought (`◐`, alone) or in need of more time (`○`, alone). The
protocol lives in `pipecat.turns.user_turn_completion_mixin`, and it only works
as well as the model follows the instructions. This suite measures that across
every text LLM service in Pipecat, so a prompt change can be checked against
the whole matrix and a new model can be scored as soon as it appears.

It is a `pipecat eval suite` like the release evals in `../release/`, with one
bot and one manifest entry per model. It differs in scale and in what it asks:
hundreds of short scripted conversations per model, each with one narrow
question of the LLM's marker and reply.

## How it works

`bot.py` is a text-only bot around one LLM service: eval transport, the user
aggregator with `FilterIncompleteUserTurnStrategies`, the LLM, the assistant
aggregator. No STT or TTS. The manifest entry's `runner_body` names the service
class, its constructor arguments, the model and its settings (the fields are in
the bot's docstring), and the bot advertises the weather and appointment tools
the tool scenarios expect unless the body says `tools: false`, for an endpoint
that rejects tool definitions; such an entry runs the scenario files without
the tool scenarios.

`scenarios/<category>.yaml` holds the scenarios of one category, sharing the
judge in `judge_text.yaml`: TypeSafe's Jev (`factory:
evals.judges.typesafe_classifier`, see `evals/judges.py`), with `gemma4:12b`
on a local Ollama as the
explainer that gives the reason for every `no` and every verdict Jev is unsure
of. Every judged reply follows a `●`, so it's a final answer, and the judge
sets `allow_continue: false`: a reply is right or wrong, never "still working
on it". To judge with Ollama alone, replace the `eval:` block with the
explainer's own. Each scenario's `context:` is the
history the assistant aggregator would have left (`● reply` entries, bare `◐`
or `○` entries, and the developer message that kicked the conversation off),
and its `turns:` carry the user's words with what the LLM must produce:

```yaml
- name: cutoff_preposition
  context:
    - role: developer
      content: The user has joined the conversation.
    - role: assistant
      content: ● If you could travel anywhere in the world right now, where would you go and why?
  turns:
    - user: I'd go to Japan because I love
      expect:
        - event: llm_marker
          marker: incomplete      # ◐ or ○; the harness records which
          marker_first: true      # nothing before the marker
          markers: 1              # one marker in the raw text
          text_after: false       # nothing after it
```

A complete turn expects `marker: complete` with `text_after: true` (a bare `●`
fails), and a `response` judged against an `eval:` criterion where the reply's
content matters. Tool scenarios expect the `function_call`, or its absence
after a cut-off request. The re-prompt scenarios send a fragment and then wait,
with no user turn, for the bot's own timeout to produce a `●` nudge.

Most scenarios exist twice: as written, and as `<name>_stt`, the same words
lowercased with punctuation stripped, since that is what a bot's LLM sees from
streaming STT and the punctuated examples in the prompt are the easy version.

## Running

```sh
evals/turn-completion/run.sh -n baseline                      # every model, every scenario
evals/turn-completion/run.sh -n baseline -p openai            # one provider
evals/turn-completion/run.sh -s cutoff/cutoff_preposition -p openai/gpt-4.1
TURN_COMPLETION_PROMPT=v3 evals/turn-completion/run.sh -n v3  # a prompt variant from prompts/
```

`run.sh` is `pipecat eval suite` over the manifest, which spawns a bot per
scenario run. Up to 32 runs go at once, taken in manifest order, so no slot
sits idle while a model still has scenarios; a full sweep is about 8,500 runs. An expectation without its own `within_ms` times out
after 30 s (`run.sh` passes `-t 30`, and a `-t` of your own overrides it),
since a model that follows the protocol answers within seconds. Entries whose
provider rate-limits (Groq, NVIDIA, Mistral Large) cap their own concurrency;
Ollama is capped at one because the judge's explainer shares it, and is
usually left out. The models' keys come from the repo's `.env`; the judge's
`TYPESAFE_API_KEY` must be exported in the shell, since the harness doesn't
read `.env`.

A run writes `test-runs/<name>/results.jsonl` (one record per scenario run: the
model, the scenario, pass or fail, each expectation and what it matched, and
for a failed run the events the bot sent, raw LLM text included) plus a bot log
and a harness trace per run under `logs/`.

## Reading the results

`results.jsonl` holds one record per run: the model (`name`), the scenario as
`<category>/<name>`, whether it passed, each expectation with what it matched
(the marker, the reply text, the tool call), the harness's failure kind and
reason, and for a failed run the events the bot sent, raw LLM text included.
The suite prints the tally and groups failures by kind; everything else is a
`jq` line over that file.

```sh
r=evals/turn-completion/test-runs/baseline/results.jsonl

# Pass rate per model
jq -r '[.name, (if .passed then 1 else 0 end)] | @tsv' $r |
  awk -F'\t' '{n[$1]++; p[$1]+=$2} END {for (m in n) printf "%5.1f%%  %4d  %s\n", 100*p[m]/n[m], n[m], m}' | sort -rn

# Pass rate per category (the scenario file), or per variant with endswith("_stt")
jq -r '[(.scenario|split("/")[0]), (if .passed then 1 else 0 end)] | @tsv' $r |
  awk -F'\t' '{n[$1]++; p[$1]+=$2} END {for (c in n) printf "%5.1f%%  %s\n", 100*p[c]/n[c], c}' | sort -n

# Failures by model and kind, and the reasons behind one kind
jq -r 'select(.passed|not) | [.name, (.failures[0].kind // "error")] | @tsv' $r | sort | uniq -c | sort -rn
jq -r 'select(.failures[0].kind=="marker_format") | [.name, .scenario, .failures[0].reason] | @tsv' $r

# Which incomplete marker the models gave where a cutoff (◐) or a request for time (○) was expected
jq -r 'select(.scenario|startswith("cutoff/")) | .turns[].expectations[] | select(.event_name=="llm_marker" and .passed) | .matched' $r | sort | uniq -c
```

The harness's kinds are the ones to read: `marker_mismatch` (the wrong
marker, its reason saying which was given and which expected), `marker_format`
(the right marker, badly formed: text before it, more than one, a bare `●`, or
text after `◐`/`○`), `judge_no`, `missing_function_call`, `unexpected_event`
for a tool call after a cut-off request, and `timeout` when no marker arrived
at all, which is also what a provider error looks like. To compare two runs,
produce the per-model table for each and `join` them, or diff the failure
lists.

## Prompt iteration

The bot uses the framework's default instructions unless
`TURN_COMPLETION_PROMPT=<name>` names a template in `prompts/`, with
`{complete}`, `{short}` and `{long}` placeholders. Run the matrix with the
default first, then the variant, and compare the two runs. The goal is to lift
the weakest models without moving the strong ones.

`prompts/` keeps the variants that have been tried. `v4.txt` is the text the
framework ships as its default; `v2.txt` and `v3.txt` are the steps that led
there, kept because their failure modes are instructive: examples written as
arrows were copied as arrows, examples written as a transcript made weak models
invent the next `User:` line, and a list of English cutoff words made models
stop recognising cutoffs in other languages. Prose examples avoided all three.
