---
name: code-review
description: Multi-agent review of a GitHub pull request for bugs, performance issues, repository convention alignment, and docstring quality. Posts validated findings as inline review comments.
disable-model-invocation: true
allowed-tools: Bash(gh issue view:*), Bash(gh search:*), Bash(gh issue list:*), Bash(gh pr comment:*), Bash(gh pr diff:*), Bash(gh pr view:*), Bash(gh pr list:*), Bash(gh pr review:*)
---

Runs a multi-agent review of a GitHub pull request and posts validated findings as inline comments.

Use the `gh` CLI to interact with GitHub. Do not use web fetch.

Create a todo list before starting.

## 1. Preflight

Launch a Haiku agent to check whether any of the following is true:

- the pull request is closed,
- the pull request is a draft,
- the pull request does not need code review (e.g. an automated PR, or a trivial change that is obviously correct),
- Claude has already commented on this PR (check `gh pr view <PR> --comments` for comments left by Claude).

If any condition is true, stop and do not proceed.

Review Claude-generated PRs as normal — being authored by Claude is not a reason to skip.

## 2. Scope the review

Read the PR with `gh pr view <PR>` and `gh pr diff <PR>`.

Record the PR's head commit SHA — findings are anchored to it, and the permalink format in step 4 requires it.

Use the PR title and description to write a one-paragraph **intent summary** describing what the PR is trying to accomplish. Do not launch an agent for this.

The **scope recipe** to hand every agent is `gh pr diff <PR>` together with the recorded head SHA, so all agents review identically-scoped changes.

## 3. Review

Read `.claude/skills/review-policy.md` and apply it, passing it the scope recipe, the changed file list, and the intent summary.

## 4. Post the review

First, assemble the complete list of comments you plan to leave and check that you are comfortable with every one. This list is for you only — do not post it anywhere.

If no findings survived validation, post a summary comment with `gh pr comment`:

```markdown
## Code review

No issues found. Checked for bugs, performance, and repository convention compliance.
```

Otherwise, post one inline comment per finding using `gh pr review`. For each comment:

- lead with the severity (**High** / **Medium** / **Low**),
- give a brief description of the issue,
- cite and link the source of any rule you invoke (e.g. the `AGENTS.md` or `CLAUDE.md` it came from),
- for small, self-contained fixes, include a committable suggestion block,
- for larger fixes (6+ lines, structural changes, or changes spanning multiple locations), describe the issue and suggested fix without a suggestion block,
- never post a committable suggestion unless committing it fixes the issue entirely. If follow-up steps are required, do not leave a suggestion block.

**Post only ONE comment per unique issue. Do not post duplicate comments.**

When linking to code, follow this format precisely, otherwise the Markdown preview won't render correctly:

`https://github.com/OWNER/REPO/blob/FULL_SHA/path/to/file.py#L10-L15`

- requires the full git SHA — command substitution such as `$(git rev-parse HEAD)` will not work, since the comment is rendered directly as Markdown,
- the repo name must match the repo being reviewed,
- `#` sign after the file name,
- line range format is `L[start]-L[end]`,
- provide at least 1 line of context before and after, centered on the line being commented on (e.g. to comment on lines 5-6, link `L4-L7`).

## 5. Report completion

Respond in chat with the number of findings posted in each category, and a link to the review. Do not paste the full review into chat.
