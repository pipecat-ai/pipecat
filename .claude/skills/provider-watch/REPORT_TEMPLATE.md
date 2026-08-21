# Report template

Every provider-watch report has this shape. The frontmatter is machine-read by
`scripts/provider-watch/digest.py`, by `publish.py`, and by the next run, so
keep the keys and value vocabularies exactly as shown. The body is for people:
short, concrete, no filler. A section with nothing to say contains the single
word `Nothing.`

A report is a snapshot of the gap between what the provider offers and what
Pipecat handles today, complete on its own.

## Frontmatter

```yaml
---
service: cartesia/tts                      # unit id from inventory.py
classes: [CartesiaTTSService, CartesiaHttpTTSService]
date: 2026-08-20
pipecat_commit: abc1234
default_model: sonic-3.5                   # what the code ships today
summary: One sentence a maintainer can act on.
models_seen:                               # sorted; from probe.py list-models or the docs
  - sonic-3
  - sonic-3.5
gaps:                                      # what Pipecat should consider; see below
  - item: Default sonic-3.5 is superseded by sonic-4 (GA 2026-08-12)
    first_seen: 2026-08-20
    action: pr                             # a prs entry covers it
  - item: Cartesia's new `emotion` controls are not reachable from Settings
    first_seen: 2026-08-06
    action: consider
    note: needs a Settings field; check whether sonic-4 voices all support it
prs:                                       # see below
  - branch: provider-watch/cartesia-tts-sonic-4
    state: branch
    summary: Default CartesiaTTSService to sonic-4
decided:                                   # things the team said not to do (or already did)
  - item: Add a `pronunciation` Settings field
    decision: wontfix — covered by pronunciation_dict_id
    source: https://github.com/pipecat-ai/provider-watch/issues/12#issuecomment-1
    date: 2026-08-13
error: null                                # or one line: why the unit could not be researched
---
```

- `gaps` is the full current list. Keep `first_seen` from the previous report when the item is the same gap, so the digest can show how long it has been open. `action` is `pr` (a `prs` entry exists for it) or `consider` (needs a maintainer's call or more work). A `note` is optional — use it for "re-check when GA" and similar.
- `prs` entries are `{branch, state: branch, summary}` for a branch you left, or `{url, state: open|merged|closed, opened, summary}` for a PR you found during dedupe. `publish.py` turns `branch` into `open` and fills `url`; `capped: true` on a branch entry means the per-run PR cap stopped it from being opened this run.
- `decided` carries forward from the previous report and grows from digest-issue comments and closed PRs. A decided item is not a gap. A decision with a revisit date ("later — revisit in Q4") becomes a gap again once the date passes.
- `error` is `null` unless the unit could not be researched (missing credential — name the variable, not the value —, provider outage, researcher failure). An errored report still lists what it could establish.

## Body

```markdown
# Cartesia TTS — 2026-08-20

One-line verdict, e.g. "Sonic 4 should replace sonic-3.5 as the default; branch ready."

## What's new for Pipecat

### PRs
- `provider-watch/cartesia-tts-sonic-4` — review: `git show provider-watch/cartesia-tts-sonic-4` — Default CartesiaTTSService to sonic-4
- https://github.com/pipecat-ai/pipecat/pull/1230 — Add sonic-4 to the sample-rate table (open since last run)

### To consider
- **Emotion controls** (since 2026-08-06) — Cartesia's `emotion` request field is not reachable from `Settings`; a field plus a pass-through in `_build_request` would do it, but check whether every sonic-4 voice supports it.

### Decided
- Add a `pronunciation` field — wontfix, covered by `pronunciation_dict_id` ([comment](https://github.com/pipecat-ai/provider-watch/issues/12#issuecomment-1), 2026-08-13)

## Verification

| model     | class              | ok | latency                          | note |
| --------- | ------------------ | -- | -------------------------------- | ---- |
| sonic-3.5 | CartesiaTTSService | ✅ | TTFB 123 ms                      |      |
| sonic-4   | CartesiaTTSService | ✅ | TTFB 118 ms                      |      |

## Sources
- https://docs.cartesia.ai/changelog — Sonic 4 GA on 2026-08-12, sonic-3.5 unchanged
- probe.py signals — cartesia 2.1.0 on PyPI (2026-08-03); cartesia-sdk.stats.yml unchanged
```

Guidance:

- Lead with what a maintainer needs to decide or review; details after. Every gap appears exactly once, under the bucket matching its `action`; decided items appear only under "Decided", in one line each.
- A branch line under "PRs" must use exactly the form ``- `<branch>` — review: `git show <branch>` — <summary>``; `publish.py` rewrites that prefix to the PR URL once the PR exists. A PR found during dedupe is listed by URL.
- The Verification table lists every probe that ran, including failures and the current default when you compared against it. Quote TTFAT (and thinking time) for LLMs and TTFB otherwise, as `probe.py` reports them.
- "Sources" is one line per page, endpoint, spec and SDK you relied on and what it told you; the next researcher starts from it.
- Never include credentials, `Authorization` headers, or raw provider error dumps that could contain them.
