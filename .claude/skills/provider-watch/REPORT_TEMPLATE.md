# Report template

Every provider-watch report has this shape. The frontmatter is machine-read by
`scripts/provider-watch/digest.py` and by the next run's delta check, so keep the
keys and value vocabularies exactly as shown. The body is for people: short,
concrete, no filler. A section with nothing to say contains the single word
`Nothing.`

## Frontmatter

```yaml
---
service: cartesia/tts                      # unit id from inventory.py
classes: [CartesiaTTSService, CartesiaHttpTTSService]
date: 2026-08-20
pipecat_commit: abc1234
default_model: sonic-3.5                   # what the code ships today
status: up-to-date                         # see vocabulary below
summary: One sentence a maintainer can act on.
models_seen:                               # sorted; from probe.py list-models or the docs
  - sonic-3
  - sonic-3.5
sources:                                   # what the delta check compares next time
  - url: https://docs.cartesia.ai/changelog
    latest_entry: "2026-08-12 — Sonic 3.5 generally available"
  - url: probe.py list-models --provider cartesia
    latest_entry: "unsupported"            # or the number of models returned
prs: []                                    # [{url, state: open|merged|closed, summary}]
open_items: []                             # noticed, not acted on; carried into the next report
---
```

`status` vocabulary:

| status            | meaning                                                                 |
| ----------------- | ----------------------------------------------------------------------- |
| `up-to-date`      | nothing new upstream that the service doesn't already handle            |
| `new-upstream`    | something new exists, nothing needs doing in Pipecat (or not yet)       |
| `prs-opened`      | at least one PR was opened (or an existing one is still open)           |
| `prs-withheld`    | a change met the PR criteria but PRs were disabled or the budget was spent |
| `needs-judgement` | a change is worth making but needs a maintainer's call or more work     |
| `blocked`         | could not research properly (no credentials, provider down, docs gone)  |
| `error`           | the researcher failed; body holds the failure                           |

## Body

```markdown
# Cartesia TTS — 2026-08-20

One-line verdict, e.g. "Sonic 3.5 is still current; nothing to do."

## What's new

### New since last report
- …

### Carried over
- … (items from the previous report's open_items or findings still relevant)

## Recommended next steps

### PRs opened, to review
- <PR URL> — what it changes and the evidence (probe results, provider statement)

### PRs withheld
- <file:line> `old` → `new` — why it meets the PR criteria, the evidence, the changelog line. Only under `--no-prs` or an exhausted PR budget; say which.

### Changes to consider
- … (what, why, and a sketch of the change or the question for the team)

## Verification

| model     | class              | ok | latency                          | note |
| --------- | ------------------ | -- | -------------------------------- | ---- |
| sonic-3.5 | CartesiaTTSService | ✅ | TTFB 123 ms                      |      |
| gpt-5.4   | OpenAILLMService   | ✅ | TTFAT 2.4 s (1.9 s thinking)     |      |

## Sources
- <url> — what it told you
```

Guidance:

- Lead with what a maintainer needs to decide or review; details after.
- "PRs withheld" holds exactly what a PR would have contained (the concrete edit, the probe table rows that justify it, the changelog line) so a test run can be checked against the PR criteria. When PRs are enabled and the budget allows, the section reads `Nothing.`
- "New since last report" is empty when this is the first report for the unit — say so in one line and put notable pre-existing gaps (e.g. a default model the provider has retired) under "Changes to consider" or open a PR.
- The Verification table lists every probe that ran, including failures and the current default when you compared against it. Quote TTFAT (and thinking time) for LLMs and TTFB otherwise, as `probe.py` reports them. Omit the section only under `--no-probe`.
- Never include credentials, `Authorization` headers, or raw provider error dumps that could contain them.
