# Review policy

Shared review policy for the `local-review` and `code-review` skills. The calling skill establishes the **review scope** and owns the output; everything between those two points is defined here.

Before applying this policy, the calling skill must have produced:

- **scope recipe** — verbatim instructions any agent can follow to reproduce the exact set of changes under review, including a pinned base commit SHA,
- **changed files** — the complete list, and
- **intent summary** — one paragraph describing what the change is trying to accomplish.

## Agent assumptions

Applies to every reviewer and validator launched below.

All tools are functional and will work without error. Do not test tools or make exploratory calls. Only invoke a tool when it is required to complete the task. Make this clear to every agent that is launched.

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
- issues that depend on specific inputs or state to manifest,
- something that appears to be a bug but is actually correct,
- future maintainability concerns,
- missing tests,
- subjective style preferences or general code quality concerns,
- pedantic nitpicks a senior engineer would not raise,
- issues a formatter or linter would already catch — do not run the linter to verify,
- issues covered by repository guidance but explicitly silenced in the code (e.g. a lint-ignore comment),
- pre-existing issues the change did not touch,
- or concerns outside the review scope unless explicitly required by repository guidance.

If you are not certain an issue is real, do not flag it. False positives erode trust and waste reviewer time.

## Severity

Assign every finding one of these levels, and use it consistently across reviewers and validators:

- **High** — will produce incorrect behavior, a crash, or a security issue; or violates a hard/unambiguous repository rule.
- **Medium** — a real but non-catastrophic issue: a concrete performance regression, or a violation of an established repository convention that isn't safety-critical.
- **Low** — a docstring or minor convention issue with no behavioral impact.

## 1. Gather context

Do this directly, without launching agents.

- Build the list of changed files, grouped by kind as makes sense for this repo (e.g. source, tests, docs, config, other).
- For every changed file, locate all applicable `AGENTS.md` and `CLAUDE.md` files in its directory hierarchy (nearest first) and collect their **contents**. A convention file applies to a given file only if it sits in that file's directory or one of its parents.
- Confirm the intent summary reflects the change as a whole. Every reviewer receives it so they can distinguish intentional behavior from defects.

## 2. Launch three reviewers in parallel

Each reviewer receives:

- the intent summary,
- the applicable `AGENTS.md` / `CLAUDE.md` contents,
- the complete list of changed files, and
- the scope recipe, with instructions to reproduce the review scope themselves so that every reviewer reviews identically-scoped changes.

Reviewers should inspect only the portions of the change relevant to their specialty. They may read additional repository context only when necessary to investigate or confirm a suspected issue.

Repository instructions from `AGENTS.md` and `CLAUDE.md` always take precedence over general engineering judgment when the two conflict.

### Reviewer 1 — Correctness (Opus)

Start with the changed code and its immediate surrounding context. Look for obvious correctness bugs introduced by the change, including:

- incorrect logic,
- off-by-one errors,
- broken conditions,
- missing cases,
- parse or runtime errors,
- security issues,
- incorrect async/concurrency usage visible locally.

Then, only where a locally-visible issue is suspected but needs confirmation, inspect additional repository context to check for issues such as:

- broken assumptions about callers or callees,
- misuse of framework or library primitives,
- incorrect handling of shared or mutable state,
- incorrect task creation,
- violations of documented architectural invariants.

Do not go looking for contextual issues that have no local symptom in the change — that's out of scope for this reviewer.

### Reviewer 2 — Repository conventions & docstrings (Sonnet)

Review the change against documented repository conventions from `AGENTS.md` and `CLAUDE.md`, including framework and library usage, class and API design, established idioms already present in this codebase, deprecation conventions, and other documented project practices — and, as part of the same pass, review new and modified public APIs for compliance with whatever docstring conventions this repo documents (missing required sections, incorrect formatting for the documented docstring style, missing or malformed deprecation directives, inaccurate documentation caused by the change).

Only report findings covered by an explicit documented rule or a well-established repository pattern.

Every finding must quote the relevant rule and name the file it came from.

### Reviewer 3 — Performance (Sonnet)

Look for concrete performance regressions introduced by the change, such as:

- unnecessary repeated work,
- inefficient algorithms,
- blocking operations in async code,
- redundant I/O,
- avoidable allocations,
- incorrect data structure choices.

Ignore pre-existing patterns that the change did not modify.

## 3. Independently validate every finding

Findings must be validated before reaching the output, but validation is batched per reviewer rather than launched per finding, to avoid re-loading the same context once per issue.

Launch one independent validator per reviewer that produced findings (so at most three, run in parallel), each receiving the full list of findings from its corresponding reviewer plus the same scope recipe given to reviewers. Match the validator's model to its reviewer's: **Opus** validates correctness findings, **Sonnet** validates convention, docstring, and performance findings.

Each validator checks every finding it receives against the actual code in a single pass. Validators must verify the code itself rather than trusting the original reviewer's description, and must judge each finding independently — one bad finding in the batch must not affect judgment of the others.

Discard findings that are:

- speculative,
- subjective,
- duplicates,
- pre-existing and untouched by the change,
- already intentionally suppressed,
- automatically handled by formatting or linting,
- or cannot be confirmed with high confidence.

For convention findings, the validator must confirm both that the quoted rule is scoped to the file in question and that it is actually violated.

## 4. Deduplicate

Merge duplicate or overlapping findings before handing them to the calling skill's output step.

If multiple reviewers identify the same underlying issue, report it only once, using the clearest explanation.

## 5. Hand off

Return the surviving findings to the calling skill, each carrying:

- a severity,
- a file path and line number,
- a description,
- a suggested fix, and
- for convention and docstring findings, the quoted rule and its source file.

The calling skill owns everything from here.
