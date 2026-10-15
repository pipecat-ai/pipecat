---
name: pr-description
description: Draft or update a GitHub PR description that explains the problem, changed behavior, and review concerns in plain language.
---

Write a pull request description for a reviewer who knows the project but hasn't
read the issue, the diff, or the development conversation. Establish why the change
exists before explaining how it works.

If the user asks for a draft, return the text without updating GitHub. Otherwise,
update the requested PR description.

## Arguments

```
/pr-description <PR_NUMBER> [--fixes <ISSUE_NUMBERS>]
```

- `PR_NUMBER` (required): The pull request to describe.
- `--fixes` (optional): Comma-separated issue numbers this PR fixes, such as `123,456`.

Examples:

- `/pr-description 3534`
- `/pr-description 3534 --fixes 123,456`

## Instructions

1. Gather the change:
   - Read the PR title, current description, base branch, and head branch.
   - Read the PR commits and diff. Use the PR's actual base and head; don't assume
     the current checkout is the PR branch or that its base is `main`.
   - Read relevant issue details, code comments, and available conversation context
     to understand the problem and the reasons for the change.
   - Collect issue numbers from `--fixes` and explicit closing references in commit
     messages. Don't infer that a related issue should be closed.

2. Check the existing description. If it's already accurate, understandable without
   outside context, and consistent with this format, leave it alone. Otherwise,
   rewrite it around the final implementation rather than appending updates.

3. Establish the core explanation before choosing bullets:
   - What does the affected component do?
   - What limitation or need prompted this change?
   - What will it do differently?
   Use only the context needed to understand this PR. Don't invent a problem to
   justify a feature or refactor.

4. Choose the distinct concerns a reviewer needs to assess. Usually these become
   3 to 6 labeled bullets; use fewer for a small change. Group by behavior or system
   responsibility, not by file, helper function, or implementation step.

5. Read the draft without consulting the diff. If the purpose or behavior is still
   unclear, supply the missing context before adding more implementation detail.

## Format

An opening paragraph, then labeled bullets. No Summary, Context, or Testing heading.
Include Breaking Changes and Fixes only when applicable.

```markdown
Pipecat groups streamed LLM text into sentences before sending it to TTS. This
replaces NLTK with `sentencex`, which includes its language rules and requires no
data downloads or tokenizer warm-up.

- **Streaming text.** Sentencex sometimes needs the complete word after a period
  to distinguish an abbreviation from a sentence ending. Pipecat waits for that
  word when needed, while sending sentences sooner when the ending is already clear.
- **Quoted abbreviations.** Sentencex can split after `"Dr.` before the rest of the
  quotation arrives. Pipecat rechecks these cases so it doesn't send an incomplete
  fragment to TTS.
- **TTS language.** Sentence grouping follows `Settings.language`, defaulting to
  English when unspecified. Language updates take effect before subsequent text
  is processed.

## Breaking Changes

- Sentence grouping can differ from Punkt and now follows the configured TTS language.
- Pipeline workers no longer emit tokenizer warm-up events.
```

### Opening paragraph

Explain what the component does, the relevant limitation or need, and what changes.
The reader should understand the purpose of the PR from this paragraph alone.
One sentence can be enough for a small change; use two or three when the reader
needs context. Don't force all three ideas into one long sentence.

Keep background directly relevant to the change. A replacement may need a brief
explanation of how the current and proposed implementations differ, especially
when that difference explains additional handling in the PR. A link to an issue
can support the explanation but must not substitute for it.

### Labeled bullets

Each bullet opens with a bolded noun label naming a class, module, surface, or
behavior. Never use essay labels such as `Motivation`, `Rationale`, `Background`,
`Context`, or `Tests`.

Lead with the behavior or problem. Explain the mechanism only when it helps the
reviewer assess the change. Usually one or two sentences suffice; a short third
sentence is fine if it supplies necessary context or an example. Don't split one
coherent concern into several bullets merely to meet a sentence limit.

When a reason is known from the issue, conversation, commit messages, or code,
include it with the relevant change. State the constraint it satisfies rather
than arguing that the decision is correct. A relevant before/after comparison is
useful; a history of attempted or rejected approaches usually isn't.

### Terminology and examples

Introduce a technical term before relying on it, or use ordinary words instead.
A label such as "lookahead" doesn't explain what text is being examined, when it
arrives, or why waiting matters. Describe those behaviors first.

For parsing, ordering, or streaming changes, prefer a short input/output example
when it makes the problem clearer than an abstract description. Examples aren't
limited to new APIs. Use inline text when sufficient; include at most one short
fenced block in the PR description when a sequence or configuration needs it.

### Breaking Changes

Include this section for changes to existing behavior, signatures, or defaults
that users need to account for. Describe the concrete effect and any required
migration. Don't omit it to meet the word target, and don't label additive features
as breaking changes solely because they are new.

### Fixes

Use `Fixes #X`, one issue per bullet, for issues this PR is intended to close.
Related PRs or issues can be linked in the opening paragraph or the relevant bullet
when the relationship helps explain scope; don't add a separate Context section.

## Length and evidence

Aim for under 200 words, but don't omit essential context to meet that target.
Cut repeated claims and secondary implementation details first. Length follows
what the reviewer needs to understand, not the size of the diff.

Describe implementation and behavior accurately against the diff. Ground problem
statements and reasons in the issue, code, or available discussion. Don't invent
motivation, generalize a narrow observation, or turn a measured sample into a
universal performance claim.

A second sentence may explain the first when that explanation adds understanding.
Remove it only when it repeats the same information without helping the reader.

Do not include:

- Claims that the change is correct or answers to objections nobody raised.
- A chronological development narrative or a paragraph weighing alternatives.
- Test counts, coverage numbers, "all tests pass", lists of tests changed, or test commands.
- A walk through the files changed.
- Aphorisms, promotional language, or a closing recap of the bullets.

## Voice

Write as if explaining the change to a teammate unfamiliar with this work. Use
plain words, contractions, and short connected sentences. Keep technical detail
where it clarifies the problem or matters to review.

Avoid forced symmetry, matched triples, "not X, but Y" constructions, and em-dash
asides where a comma or another sentence would work. Don't sacrifice a clear
explanation to make every bullet the same shape or length.

## Checklist

- [ ] The opening establishes enough context to explain the purpose of the PR.
- [ ] A reader can describe the relevant limitation and new behavior without opening the diff.
- [ ] Bullets cover distinct review concerns and lead with behavior or problems.
- [ ] Technical terms are introduced, and difficult edge cases have an example where useful.
- [ ] Reasons and before/after comparisons are grounded in available evidence.
- [ ] The description reflects the final implementation, not the development history.
- [ ] Breaking changes and closing issue references are included when applicable.
- [ ] No Testing section, test counts, coverage numbers, or redundant recap.
- [ ] The description aims for under 200 words without withholding necessary context.
