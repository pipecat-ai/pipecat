# Report template

Every provider-watch report has this shape. The frontmatter is machine-read by
`scripts/provider-watch/digest.py`, by `publish.py`, and by the next run's delta
check, so keep the keys and value vocabularies exactly as shown. The body is for
people: short, concrete, no filler. A section with nothing to say contains the
single word `Nothing.`

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
prs: []                                    # see below
open_items: []                             # noticed, not acted on; carried into the next report
---
```

`status` vocabulary:

| status            | meaning                                                                 |
| ----------------- | ----------------------------------------------------------------------- |
| `up-to-date`      | nothing new upstream that the service doesn't already handle            |
| `new-upstream`    | something new exists, nothing needs doing in Pipecat (or not yet)       |
| `pr-proposed`     | a branch was left for a PR, or an existing PR for this unit is open     |
| `needs-judgement` | a change is worth making but needs a maintainer's call or more work     |
| `blocked`         | could not research properly (no credentials, provider down, docs gone)  |
| `error`           | the researcher failed; body holds the failure                           |

`prs` entries:

```yaml
prs:
  - branch: provider-watch/cartesia-tts-sonic-4   # what the researcher writes
    state: branch
    summary: Default CartesiaTTSService to sonic-4
  - url: https://github.com/pipecat-ai/pipecat/pull/1234   # after publish.py opens it,
    state: open                                             # or an existing PR found during dedupe
    opened: 2026-08-20
    summary: Default CartesiaTTSService to sonic-4
```

`state` is `branch` (committed locally, not yet a PR), `open`, `merged`, or `closed`. `publish.py` turns `branch` into `open` and fills `url`; `capped: true` on a branch entry means the per-run PR cap stopped it from being opened this run.

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

### PRs
- `provider-watch/cartesia-tts-sonic-4` — review: `git show provider-watch/cartesia-tts-sonic-4` — Default CartesiaTTSService to sonic-4
- https://github.com/pipecat-ai/pipecat/pull/1230 — Add sonic-4 to the sample-rate table (open since last run)

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
- A branch line under "PRs" must use exactly the form ``- `<branch>` — review: `git show <branch>` — <summary>``; `publish.py` rewrites that prefix to the PR URL once the PR exists. A PR found during dedupe is listed by URL.
- "New since last report" is empty when this is the first report for the unit — say so in one line and put notable pre-existing gaps (e.g. a default model the provider has retired) under "Changes to consider" or propose a PR.
- The Verification table lists every probe that ran, including failures and the current default when you compared against it. Quote TTFAT (and thinking time) for LLMs and TTFB otherwise, as `probe.py` reports them.
- Never include credentials, `Authorization` headers, or raw provider error dumps that could contain them.
