---
name: local-review
description: Multi-agent review of the current branch's local changes (uncommitted + outgoing commits) for bugs, performance issues, Pipecat/CLAUDE.md style alignment, and docstring quality. Writes findings to .local_review.md instead of applying fixes or posting comments.
disable-model-invocation: true
---

Produce a written self-review of the changes on the current local branch.

This skill exists between the two other review skills in this repo:

- `cleanup` reviews the branch and directly *applies* readability/docstring/pattern fixes in one pass, with no independent validation.
- `code-review` reviews a **GitHub PR** with a multi-agent pipeline (parallel specialists + a validation pass per issue) and posts inline PR comments.

`local-review` borrows `code-review`'s multi-agent-plus-validation architecture (for high signal / low false-positive rate) but points it at the **local branch diff** instead of a PR, expands categories to match `cleanup`'s concerns (style, performance, docstrings) alongside bugs, and never edits code or talks to GitHub — it only writes a markdown report to `.local_review.md` at the repo root for you to act on yourself.

**Agent assumptions (applies to all agents and subagents):** all tools are functional and will work without error — do not test tools or make exploratory calls. Only call a tool if it's required to complete the task.

## Steps

1. **Scope the diff.** Determine what "the current branch's changes" means:
   - `git merge-base main HEAD` (fall back to `origin/main` if `main` doesn't exist locally) to find the base.
   - Committed changes: `git diff <base>...HEAD`.
   - Working tree changes: `git diff` (unstaged) and `git diff --cached` (staged).
   - New files: `git status --porcelain` for untracked files, then read their contents — they're part of the branch's work even though `git diff` won't show them.

   If there is no diff at all (clean branch, nothing ahead of base), stop and tell the user there's nothing to review.

2. **Gather context**, without spinning up subagents for it (this is cheap, do it directly):
   - List of changed files, grouped by kind (services, examples, tests, processors, other).
   - Every `CLAUDE.md`/`AGENTS.md` that shares a path with a changed file or its parent directories.
   - A one-paragraph read of the diff to understand intent (what is this branch trying to do) — this gets passed to every reviewer agent below so they can tell "intentional" from "bug."

3. **Launch 5 specialist agents in parallel**, each given: the full diff, the one-paragraph intent summary, and the relevant `CLAUDE.md`/`AGENTS.md` paths (content, not just paths). Each returns a list of issues (file, line range, description, suggested fix). Instruct every agent that CLAUDE.md/AGENTS.md text takes precedence over general judgment when the two conflict.

   - **Agent 1 — Opus bug agent (diff-only).** Scan strictly the diff for obvious correctness bugs: wrong logic, off-by-one, unhandled cases visible in the diff itself, without reading outside context.
   - **Agent 2 — Opus bug agent (contextual).** Look for problems in the introduced code that require repo context to see: broken assumptions about callers, misuse of framework primitives (e.g. `self.create_task` vs raw `asyncio.create_task`, frame push direction, uninterruptible frame handling), security issues.
   - **Agent 3 — Sonnet style/CLAUDE.md agent.** Check Pipecat pattern consistency per `AGENTS.md` and `cleanup`'s known patterns: correct service base-class inheritance, constructor conventions, frame emission direction, metrics hooks (`can_generate_metrics`, TTFB/TTFA), deprecation-marker conventions, dataclass-vs-Pydantic usage, example structure (`examples/07-interruptible.py` as reference). Only flag things a specific CLAUDE.md/AGENTS.md rule or an established repo pattern actually covers — cite the rule.
   - **Agent 4 — Sonnet performance agent.** Conservative, non-hypothetical performance issues introduced by the diff: inefficient loops/repeated work, wrong data structure, blocking calls in async code, redundant I/O. Skip anything that's a pre-existing pattern the diff didn't touch.
   - **Agent 5 — Sonnet docstring agent.** Google-style docstring completeness/correctness per `AGENTS.md`'s docstring conventions: missing `Args:`/`Parameters:`/`Returns:` sections on new or changed public classes/methods, wrong style for dataclass vs Pydantic, deprecation directives missing the `.. deprecated::` block or not leading with the replacement reference.

   **High-signal bar for all 5 agents** (same bar as `code-review`): flag only issues you're confident are real — code that will fail to run/parse, will definitely produce wrong results, a performance regression with a concrete trigger, or a docstring/style rule you can quote and point at. Do not flag style nitpicks a linter would catch, subjective preferences, or anything that depends on inputs/state you can't verify from the diff.

4. **Validate every flagged issue** with a fresh subagent per issue (Opus for Agent 1/2 bug findings, Sonnet for Agent 3/4/5 findings), passing the intent summary and the specific issue. The validator's only job: confirm with high confidence that the issue is real by checking the actual code (not just trusting the description). Drop anything that doesn't validate.

   Apply this shared false-positive filter (do NOT flag / validate away):
   - Pre-existing issues untouched by this diff.
   - Something that looks like a bug but is actually correct.
   - Pedantic nitpicks a senior engineer wouldn't raise.
   - Issues a linter/formatter would already catch.
   - General code-quality or test-coverage concerns not required by CLAUDE.md/AGENTS.md.
   - Issues explicitly silenced in code (e.g. a lint-ignore comment).

5. **Write `.local_review.md`** at the repo root (overwrite if it exists), with this structure:

   ```markdown
   # Local Review — <branch name>

   Reviewed <base>..HEAD plus uncommitted changes, generated <date>.

   ## Summary
   <1-3 sentence read of what the branch does>

   ## Bugs & Correctness
   - [ ] `path/to/file.py:123` — <description>. Suggested fix: <fix or "see below">

   ## Performance
   - [ ] ...

   ## Pipecat Style / CLAUDE.md Alignment
   - [ ] `path/to/file.py:45` — <description>. Rule: <quoted CLAUDE.md/AGENTS.md line, with path>

   ## Docstrings
   - [ ] ...
   ```

   Omit any section with zero validated findings rather than leaving it empty. If nothing validated in any category, still write the file with just the Summary and a one-line "No issues found" note.

6. **Make sure `.local_review.md` is gitignored.** If `.gitignore` doesn't already have an entry for it, add one (under a `# Local review output` comment) so the report never gets committed or shows up in `git status` noise.

7. Report back to the user in the chat: a short summary of counts per category and the path to the file. Do not paste the full report into the chat — the user reads `.local_review.md` directly.

## Notes

- This skill never edits reviewed code and never touches GitHub — it is strictly read + report.
- Re-running overwrites `.local_review.md`, so it's safe to invoke repeatedly as you iterate on the branch.
- If `main`/`origin/main` can't be resolved as a base, ask the user which branch to diff against rather than guessing.
