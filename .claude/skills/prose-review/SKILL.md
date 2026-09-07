---
name: prose-review
description: Check that added comments, docstrings, and changelog entries document the code rather than narrate the change that produced it
argument-hint: "[staged|branch|<path>] (default: branch)"
---

Review the prose this work adds — comments, docstrings, changelog entries — against
the "Writing for Future Readers" section of `AGENTS.md`, which is the authority.
This skill is the operational check.

The reader you are writing for has never seen this change. They open the file
months from now and see only the code as it stands. Detail that feels important
while making a change — what the code used to do, why an alternative was
rejected, that the fix was verified — is invisible context to them, and often
actively misleading.

## Scope

| Argument | Diff to review |
| --- | --- |
| `staged` | `git diff --cached -U0` |
| `branch` (default) | `git diff $(git merge-base main HEAD) -U0` |
| `<path>` | `git diff $(git merge-base main HEAD) -U0 -- <path>` |

The `branch` and `<path>` scopes diff against the merge base, so they cover
committed *and* uncommitted work — prose written moments ago still needs review.

Review **added lines only** (`^\+`): comments, docstrings, and any
`changelog/*.md` content. Untouched prose is out of scope. Read the surrounding
code for each hit — the judgment depends on what the code already shows.

## The Test

For each added line, ask: **does this describe the code as it now stands, or does
it narrate the change that produced it?**

Rewrite or delete anything that:

- Describes what the code used to do, or contrasts old behavior with new
- Argues the change is correct, or records that it was tested or verified
- Names an alternative that was considered and rejected
- Only restates what the adjacent code already shows
- Uses shorthand that made sense mid-task but won't to someone reading cold

Rationale survives only when its absence would puzzle a future reader — a
constraint or non-obvious decision the code itself cannot express. Keep those,
and prefer stating the constraint over recounting the discovery.

## Calibration

Do not pattern-match on words like "previously", "instead of", "no longer", or
"used to". They appear constantly in correct prose. Judge the sentence's subject:
the code, or the change.

Legitimate — all describe current behavior:

```python
# This queue is the queue used to push frames to the pipeline.
# ...discarded rather than flushed, since it's no longer wanted.
# ...at arg-parse time so users get a clear message instead of a downstream crash.
```

Violations, and their fixes:

```python
# We used to buffer here, but that broke interruptions, so now we flush.
# → Flush immediately; buffering here delays interruption handling.

# Changed from a list to a deque for O(1) pops.
# → delete; the code shows the deque

# Note: verified this against the Deepgram sandbox.
# → delete

# Tried a lock here first, but it deadlocked with the task manager.
# → Reentrant by design: the task manager may re-enter during cancellation.
```

```python
"""Replay buffered word events, replacing the old single-slot matcher."""
# → """Replay buffered word events now that a new slot may match them."""
```

## Changelog Exception

Changelog entries and PR descriptions are release notes for users, so contrasting
old and new *user-visible behavior* belongs there — `/changelog` models this
deliberately. What does not belong is development-process detail: rejected
approaches, review back-and-forth, verification narrative, or refactoring
described from the inside.

## Remediation

Fix the prose directly, then land the correction at the cheapest point:

| Situation | Fix |
| --- | --- |
| Nothing committed yet | Just edit the files |
| Committed, not pushed, one commit | Edit, then `git commit --amend --no-edit` |
| Committed, not pushed, several commits | Edit, commit the fixes, then `/squash-commits` |
| Commit message wording, not pushed | `git commit --amend -m` for the tip; `/squash-commits` across the branch (interactive rebase is unavailable) |
| Pushed, no PR yet | Fix as above, then force-push with `--force-with-lease` |
| Changelog file, any time | Edit and commit normally — no history rewrite |
| PR description, any time | `gh pr edit <pr_number> --body` |
| PR already under review | Follow-up commit; do not rewrite pushed history |

## Report

List each violation as `file:line`, the offending text, and what it became. If
the prose is clean, say so explicitly — callers rely on knowing the check ran.
