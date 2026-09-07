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
    priority: medium                       # high | medium | low; see below
    note: a field plus a pass-through in `_build_request` would do it
    needs: not every sonic-4 voice supports `emotion`; exposing it needs a stance on unsupported voices
prs:                                       # see below
  - branch: provider-watch/cartesia-tts-sonic-4
    state: branch
    summary: Default CartesiaTTSService to sonic-4
error: null                                # or one line: why the unit could not be researched
---
```

- `gaps` is the full current list. Keep `first_seen` from the previous report when the item is the same gap, so the digest can show how long it has been open. `action` is `pr` (a `prs` entry exists for it) or `consider` (needs a maintainer's call or more work). Every `consider` item carries a `needs` — one line naming the decision or unknown that keeps it out of an automatic PR. A `note` is optional — the what and the evidence, "re-check when GA" and similar; `needs` is the question put to the maintainer.
- Every `consider` item carries a `priority`, which orders the digest:
  - `high` — users are affected now or imminently: a request the provider rejects or has scheduled to reject (a deprecated parameter with a removal date), a crash or hang path, a default that is retiring, something a released model needs in order to work at all.
  - `medium` — a capability users plausibly want that the service cannot express: a missing `Settings` field, a model an allowlist blocks, an SDK pin behind the provider's current major.
  - `low` — hygiene: naming, docs that live in another repo, preview-only models to re-check, enum cleanup.
- `prs` entries are `{branch, state: branch, summary}` for a branch you left, or `{url, state: open|merged|closed, opened, summary}` for a PR you found during dedupe. The `summary` covers everything on the branch; when the branch has more than one commit it becomes the PR title. `publish.py` turns `branch` into `open` and fills `url`; `capped: true` on a branch entry means the per-run PR cap stopped it from being opened this run.
- An item covered by a decision in force (see "The decisions file" below) is not a gap and does not appear in the report.
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
- **Emotion controls** (medium, since 2026-08-06) — Cartesia's `emotion` request field is not reachable from `Settings`; a field plus a pass-through in `_build_request` would do it. — *Needs a call: not every sonic-4 voice supports `emotion`, so exposing it needs a stance on unsupported voices.*

## Verification

| model     | class              | ok | latency                          | note |
| --------- | ------------------ | -- | -------------------------------- | ---- |
| sonic-3.5 | CartesiaTTSService | ✅ | TTFB 123 ms                      |      |
| sonic-4   | CartesiaTTSService | ✅ | TTFB 118 ms                      |      |

## Sources
- https://docs.cartesia.ai/changelog — Sonic 4 GA on 2026-08-12, sonic-3.5 unchanged
- probe.py sdk-versions — cartesia 2.1.0 on PyPI (2026-08-03), inside our pin
```

Guidance:

- Lead with what a maintainer needs to decide or review; details after. Every gap appears exactly once, under the bucket matching its `action`, "To consider" items ordered high → medium → low with the priority in the lead-in and the *Needs a call:* line closing each.
- A branch line under "PRs" must use exactly the form ``- `<branch>` — review: `git show <branch>` — <summary>``; `publish.py` rewrites that prefix to the PR URL once the PR exists. A PR found during dedupe is listed by URL.
- The Verification table lists every probe that ran, including failures and the current default when you compared against it. Quote TTFAT (and thinking time) for LLMs and TTFB otherwise, as `probe.py` reports them; for a comparison that decided anything, quote medians with the repeat count (e.g. `TTFAT 998 ms (median of 5)`).
- "Sources" is one line per page, endpoint, spec and SDK you relied on and what it told you; the next researcher starts from it.
- Frontmatter is YAML: quote any value that starts with a backtick, `*`, `&`, `[`, `{`, `#`, `|`, `>`, `%`, `@` or contains `: ` — e.g. `note: "`extra` covers it"` — or the report fails to parse and is listed as an error.
- Never include credentials, `Authorization` headers, or raw provider error dumps that could contain them.

## The decisions file

`reports/<provider>/<unit-suffix>/decisions.md` sits beside the unit's dated reports and holds
the decisions currently in force — the reason a known gap is not listed in the report. One
bullet per decision:

```markdown
# Decisions — cartesia/tts

- Add a `pronunciation` Settings field — wontfix, covered by `pronunciation_dict_id` ([comment](https://github.com/pipecat-ai/provider-watch-reports/issues/12#issuecomment-1), 2026-08-13)
- Expose `flush_id` — later, revisit after 2026-11-01 ([comment](https://github.com/pipecat-ai/provider-watch-reports/issues/14#issuecomment-2), 2026-08-20)
```

- Append an entry when a digest-issue comment or a closed-unmerged PR decides an item: the item
  as the report named it, the decision in the commenter's words, and the comment or PR as the
  linked source, with the date.
- Delete an entry when it stops being in force — its revisit date has passed (the item is a gap
  again) or a code change made it moot. Git history is the archive; the file holds only what
  currently applies.
