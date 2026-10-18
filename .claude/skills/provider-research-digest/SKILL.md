---
name: provider-research-digest
description: Render the digest for one provider-research date from every report on disk, topped with authored highlight bullets; publishing is scripts/provider-watch/publish.py's job, run outside this skill
disable-model-invocation: true
argument-hint: "[--date YYYY-MM-DD]"
---

Render `digests/<date>.md` in the reports checkout from every report carrying the date, topped with highlight bullets you author. The digest is a function of the reports on disk, not of any one research run — re-running this skill after further same-date research replaces the digest with a fresh view of the whole date. Everything stays local — this skill publishes nothing; pushing the digest and opening the digest issue are `scripts/provider-watch/publish.py --finalize`'s job, run afterwards by whoever invoked this.

## Arguments

```
/provider-research-digest [--date YYYY-MM-DD]
```

- `--date YYYY-MM-DD` — the date to digest. Defaults to today.

## Instructions

### Step 1: Sync the reports checkout

Record `RUN_DATE` as `--date` if given, else today's date (`YYYY-MM-DD`), and pick a scratch directory outside the repo (your session scratchpad if you have one, else `mktemp -d -t provider-research-digest`). The reports checkout is always `./_reports` in this repo (gitignored). If it is missing, `gh repo clone pipecat-ai/provider-watch-reports _reports`; if it exists and has a remote, `git -C _reports pull --ff-only` — the digest must see every report published for the date, not a stale checkout. Stop with a clear error if no `_reports/reports/*/*/<RUN_DATE>.md` exists.

### Step 2: Render a draft

```bash
uv run python scripts/provider-watch/digest.py --reports _reports --date <RUN_DATE> --out <scratch>/digest-draft.md
```

Read the draft: it aggregates every unit's summary, PRs, changes to consider and errors for the date.

### Step 3: Author the highlights

Write up to 5 highlight bullets to `<scratch>/highlights.md` from the draft: what a maintainer should look at first — services broken out of the box, PRs and branches worth reviewing first, long-open gaps that finally moved, providers that errored. Judge from the whole date, not from whichever units were researched most recently, and skip bullets when nothing stands out. Match the draft's conventions (unit ids and code in backticks, report paths where a pointer helps).

### Step 4: Render the final digest and hand off

1. ```bash
   uv run python scripts/provider-watch/digest.py --reports _reports --date <RUN_DATE> --highlights <scratch>/highlights.md --out _reports/digests/<RUN_DATE>.md
   ```
2. Print the digest path and end with the next step, which belongs to the invoker, not to you — print the command together with its explanation, and never run it:
   - `uv run python scripts/provider-watch/publish.py --date <RUN_DATE> --finalize` — publishes everything on disk for the date, digest included: pushes branches, reports and the digest, and opens (or updates) the digest issue.

## Guardrails

- Never print, commit, or paste environment variable values, `Authorization` headers, or raw API keys.
- This skill publishes nothing: never push, never open PRs or issues, never run `publish.py` — print its command instead.
- The digest body comes from `digest.py`; author only the highlights, and do not hand-edit the rendered sections.
