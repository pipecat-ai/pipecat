---
name: local-review
description: Multi-agent review of the current branch's local changes (uncommitted + outgoing commits) for bugs, performance issues, repository convention alignment, and docstring quality. Writes findings to .local_review.md instead of applying fixes or posting comments.
disable-model-invocation: true
---

Runs a multi-agent review of the current branch's local changes and writes findings to `.local_review.md`. Read-only — it never edits code, commits, or touches GitHub.

Create a todo list before starting.

## 1. Scope the review

Determine what "the current branch's changes" means.

- Compute the merge base against the repo's main branch: try `git merge-base main HEAD`, then `master`, then `origin/main`, then `origin/master`.
- If none can be resolved, stop and ask the user which branch should be used as the comparison base.

Record the resolved base commit SHA. Collect:

- committed changes using `git diff <base>...HEAD`
- unstaged changes using `git diff`
- staged changes using `git diff --cached`
- untracked files via `git status --porcelain`, reading their contents since they are part of the branch even though they do not appear in `git diff`

If there are no committed, staged, unstaged, or untracked changes, stop and tell the user there is nothing to review.

Read enough of the changes to write a one-paragraph **intent summary** describing what the branch is trying to accomplish.

The **scope recipe** to hand every agent is the four collection commands above, with `<base>` replaced by the resolved SHA so all agents review identically-scoped changes.

## 2. Review

Read `.claude/skills/review-policy.md` and apply it, passing it the scope recipe, the changed file list, and the intent summary.

## 3. Write `.local_review.md`

Overwrite the file if it already exists. Use the following structure:

```markdown
# Local Review — <branch name>

Reviewed <base>..HEAD plus uncommitted changes, generated <date>.

## Summary

<1–3 sentence summary describing the branch>

## Bugs & Correctness

- [ ] **High** — `path/to/file:123`

  Description.

  Suggested fix: ...

## Performance

- [ ] **Medium** — `path/to/file:87`

  Description.

  Suggested fix: ...

## Repository Conventions & Docstrings

- [ ] **Low** — `path/to/file:45`

  Description.

  Rule:

  > "<quoted rule>"

  Source: `path/to/AGENTS.md`
```

Omit any section with zero validated findings.

If no findings remain after validation, still write the report containing the Summary and a single line stating **"No issues found."**

## 4. Report completion

Respond in chat with the number of findings in each category and the location of `.local_review.md`. Do **not** paste the report into chat.

If `.local_review.md` is not ignored by Git, suggest adding it to `.gitignore`, but do not modify the repository.

## Notes

- This skill never edits reviewed code.
- This skill never commits changes.
- This skill never interacts with GitHub.
- The only file it writes is `.local_review.md`.
- Re-running the skill overwrites the report, making it safe to use repeatedly while iterating on a branch.
