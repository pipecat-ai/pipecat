---
name: pr-description
description: Update a GitHub PR description with a summary of changes
---

Update a GitHub pull request description based on the changes in the PR.

## Arguments

```
/pr-description <PR_NUMBER> [--fixes <ISSUE_NUMBERS>]
```

- `PR_NUMBER` (required): The pull request number to update
- `--fixes` (optional): Comma-separated issue numbers this PR fixes (e.g., `--fixes 123,456`)

Examples:
- `/pr-description 3534`
- `/pr-description 3534 --fixes 123,456`

## Instructions

1. Gather the change:
   - GitHub plugin for PR details (title, current description, base branch)
   - `git log main..HEAD --oneline` and `git diff main..HEAD`
   - Issue numbers from `--fixes` and from commit messages (`Fixes #123`, `Closes #456`)

2. Check the existing description. If it is already complete and accurate, do
   nothing. Update it if it is missing sections, outdated, or still the template
   placeholder.

3. Decide the bullets before writing any. List the distinct things a reviewer has
   to check — usually 3 to 6, rarely more than 8. One bullet each. A concern that
   needs more than two sentences is two concerns.

## Format

An opening sentence, then labeled bullets. Headings only where shown.

```markdown
A scenario can now be scripted or simulated.

- **Simulated scenarios.** `persona:` plus `goal:` lets an LLM play the caller and
  hang up with an `end_call` tool. A judge reads the transcript and decides
  `success:`.
- **Metrics.** A judged `criterion` is decided per bot turn, scored as the share of
  turns that passed. A `measure` is computed from the run, no judge.
- **Runs.** `runs: N` and every run must pass. `--repeat` sweeps only measure.
- **CLI.** `pipecat eval run` takes either kind; `-k simulation` selects them in a
  suite.

## Breaking Changes

- `EvalSession.run()` returns `EvalScriptResult` or `EvalSimulationResult`
- `RTVIEvalSerializer` is now `EvalSerializer` — the old name is a deprecated alias

## Fixes

- Fixes #123
```

- **Opening sentence.** What is different now, in one line, no heading. A reviewer
  who reads only this should know what the PR is.
- **Bullets.** Each opens with a bolded noun label naming a part of the system — a
  class, a module, a surface, a behavior — so a reviewer can skip to the part they
  own. Never label a bullet with a section of an essay: `Motivation`, `Rationale`,
  `Background`, `Context`, `Tests`. Two sentences at most.
- **Example.** At most one short fenced block, and only when the PR adds an API or
  config surface a reviewer would otherwise have to reconstruct.
- **Breaking Changes.** Include whenever behavior, signatures, or defaults change
  for existing users. Never omit this section to save space.
- **Fixes.** Use `Fixes #X`, one per line, so GitHub closes them on merge.

Nothing else. No Testing section, no Context section. A PR that belongs to a series
says so in a trailing clause on the opening sentence.

## Rules

Length follows the reviewer, not the diff. Aim under 200 words. A small diff gets a
small description.

Every bullet is a fact a reviewer can check against the diff. Give a reason only
where a reviewer would otherwise ask "why would you do that?", and give it as a
clause rather than a bullet. `AGENTS.md` sets the same standard for comments, under
"Writing for Future Readers".

Never include:

- A sentence arguing the change is correct
- A second sentence that elaborates the first rather than adding a fact
- Alternatives considered and rejected
- Test counts, coverage numbers, "all tests pass", which tests changed, or how to
  run the tests
- Aphorism, metaphor where a plain noun works, or a fact restated more elegantly
- A walk through the files changed

## Checklist

- [ ] Opening sentence says what is different now
- [ ] Every bullet has a noun label and is two sentences or fewer
- [ ] No sentence argues, and no rejected alternatives
- [ ] No Testing section, test counts, or coverage numbers
- [ ] Breaking changes documented, if any
- [ ] Under 200 words
- [ ] A reviewer who has not opened the diff can say what to review
