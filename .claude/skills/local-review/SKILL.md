---
name: local-review
description: Multi-agent review of the current branch's local changes (uncommitted + outgoing commits) for bugs, performance issues, Pipecat/CLAUDE.md style alignment, and docstring quality. Writes findings to .local_review.md instead of applying fixes or posting comments.
disable-model-invocation: true
---

Produce a written self-review of the changes on the current local branch.

This skill exists between the two other review skills in this repo:

- `cleanup` reviews the branch and directly *applies* readability, docstring, and repository-pattern fixes in one pass, with no independent validation.
- `code-review` reviews a **GitHub PR** with a multi-agent pipeline (parallel specialists followed by independent validation) and posts inline PR comments.

`local-review` borrows `code-review`'s multi-agent architecture but points it at the **local branch diff** instead of a PR. It expands the review to cover correctness, performance, repository conventions, and docstrings while remaining strictly read-only: it never edits reviewed code, commits changes, or interacts with GitHub. It writes its findings to `.local_review.md` for the developer to review.

**Agent assumptions (applies to all reviewers and validators):** all tools are functional and will work without error. Do not test tools or make exploratory calls. Only invoke a tool when it is required to complete the task.

## Review philosophy

Review the code as though you are the final approving reviewer.

Prefer missing a questionable issue over reporting a false positive.

Only report findings that you are confident an experienced reviewer would raise because they are:

- a correctness bug,
- a concrete performance regression,
- a documented repository convention violation, or
- a docstring issue required by `AGENTS.md` or `CLAUDE.md`.

Do **not** report:

- speculative or hypothetical bugs,
- future maintainability concerns,
- missing tests,
- subjective style preferences,
- issues a formatter or linter would already catch,
- or concerns outside the scope of the current diff unless explicitly required by repository guidance.

## Severity

Assign every finding one of these levels, and use it consistently across reviewers and validators:

- **High** — will produce incorrect behavior, a crash, or a security issue; or violates a hard/unambiguous repository rule.
- **Medium** — a real but non-catastrophic issue: a concrete performance regression, or a violation of an established repository convention that isn't safety-critical.
- **Low** — a docstring or minor convention issue with no behavioral impact.

## Steps

### 1. Scope the review

Determine what "the current branch's changes" means.

- Compute the merge base using `git merge-base main HEAD`.
- If `main` does not exist locally, fall back to `origin/main`.
- If neither can be resolved, stop and ask the user which branch should be used as the comparison base.

Record the resolved base commit SHA — every reviewer in step 3 must scope its diff to this exact SHA so all three agree on what "the diff" is.

Collect:

- committed changes using `git diff <base>...HEAD`
- unstaged changes using `git diff`
- staged changes using `git diff --cached`
- untracked files via `git status --porcelain`, reading their contents since they are part of the branch even though they do not appear in `git diff`

If there are no committed, staged, unstaged, or untracked changes, stop and tell the user there is nothing to review.

### 2. Gather context

Do this directly without launching reviewer agents.

- Build the list of changed files, grouped by kind (services, examples, tests, processors, other).
- For every changed file, locate all applicable `AGENTS.md` and `CLAUDE.md` files in its directory hierarchy (nearest first) and collect their contents.
- Read enough of the diff to produce a one-paragraph summary describing the intent of the branch. This summary will be passed to every reviewer so they can distinguish intentional behavior from defects.

### 3. Launch three Sonnet reviewer agents in parallel

Each reviewer receives:

- the branch intent summary,
- the applicable `AGENTS.md` / `CLAUDE.md` contents,
- the complete list of changed files,
- the resolved base commit SHA from step 1, with instructions to reproduce the exact same diff scope themselves (`git diff <base>...HEAD`, plus staged, unstaged, and untracked changes) — every reviewer must review identically-scoped changes.

Reviewers should inspect only the portions of the diff relevant to their specialty. They may read additional repository context only when necessary to investigate or confirm a suspected issue.

Repository instructions from `AGENTS.md` and `CLAUDE.md` always take precedence over general engineering judgment when the two conflict.

#### Reviewer 1 — Correctness

Start with the changed code and its immediate surrounding context. Look for obvious correctness bugs introduced by the diff, including:

- incorrect logic,
- off-by-one errors,
- broken conditions,
- missing cases,
- parse or runtime errors,
- incorrect async usage visible locally.

Then, only where a locally-visible issue is suspected but needs confirmation, inspect additional repository context to check for issues such as:

- broken assumptions about callers or callees,
- misuse of framework primitives,
- incorrect processor behavior,
- incorrect task creation,
- frame direction mistakes,
- security issues,
- violations of framework invariants.

Do not go looking for contextual issues that have no local symptom in the diff — that's out of scope for this reviewer.

#### Reviewer 2 — Repository Conventions & Docstrings

Review the diff against documented repository conventions from `AGENTS.md` and `CLAUDE.md`, including framework usage, inheritance, constructors, metrics hooks, frame handling, dataclass/Pydantic conventions, deprecation conventions, example structure, and other documented project practices — and, as part of the same pass, review new and modified public APIs for compliance with the repository's documented docstring conventions (missing required sections, incorrect Google-style formatting, incorrect dataclass vs. Pydantic documentation, missing or malformed deprecation directives, inaccurate documentation caused by the current diff).

Only report findings covered by an explicit documented rule or a well-established repository pattern.

Every finding must quote the relevant rule.

#### Reviewer 3 — Performance

Look for concrete performance regressions introduced by the diff, such as:

- unnecessary repeated work,
- inefficient algorithms,
- blocking operations in async code,
- redundant I/O,
- avoidable allocations,
- incorrect data structure choices.

Ignore pre-existing patterns that the diff did not modify.

### 4. Independently validate every finding

Findings must be validated before being included in the report, but validation is batched per reviewer rather than launched per finding, to avoid re-loading the same context once per issue: launch one independent Sonnet validator per reviewer from step 3 (so at most three validator agents, run in parallel), and give each validator the full list of findings from its corresponding reviewer plus the same base SHA / diff-scoping instructions from step 3. Each validator checks every finding it receives against the actual code in a single pass.

Validators must verify the actual code rather than trusting the original reviewer's description, and must judge each finding independently — one bad finding in the batch should not affect judgment of the others.

Discard findings that are:

- speculative,
- subjective,
- duplicates,
- pre-existing and untouched by the diff,
- already intentionally suppressed,
- automatically handled by formatting or linting,
- or cannot be confirmed with high confidence.

### 5. Deduplicate findings

Merge duplicate or overlapping findings before writing the report.

If multiple reviewers identify the same underlying issue, report it only once using the clearest explanation.

### 6. Write `.local_review.md`

Overwrite the file if it already exists.

Use the following structure:

```markdown
# Local Review — <branch name>

Reviewed <base>..HEAD plus uncommitted changes, generated <date>.

## Summary

<1–3 sentence summary describing the branch>

## Bugs & Correctness

- [ ] **High** — `path/to/file.py:123`

  Description.

  Suggested fix: ...

## Performance

- [ ] **Medium** — `path/to/file.py:87`

  Description.

  Suggested fix: ...

## Repository Conventions & Docstrings

- [ ] **Low** — `path/to/file.py:45`

  Description.

  Rule:

  > "<quoted rule>"

  Source: `path/to/AGENTS.md`
```

Omit any section with zero validated findings.

If no findings remain after validation, still write the report containing:

- the Summary, and
- a single line stating **"No issues found."**

### 7. Report completion

Respond in chat with:

- the number of findings in each category,
- the location of `.local_review.md`.

Do **not** paste the report into chat.

If `.local_review.md` is not ignored by Git, suggest adding it to `.gitignore`, but do not modify the repository.

## Notes

- This skill never edits reviewed code.
- This skill never commits changes.
- This skill never interacts with GitHub.
- The only file it writes is `.local_review.md`.
- Re-running the skill overwrites the report, making it safe to use repeatedly while iterating on a branch.
