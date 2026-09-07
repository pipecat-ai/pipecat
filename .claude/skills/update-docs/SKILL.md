---
name: update-docs
description: Update documentation pages to match source code changes on the current branch
---

Update documentation pages to reflect source code changes on the current branch. Analyzes the diff against main, maps changed source files to their corresponding doc pages, and makes targeted edits.

This skill is repo-agnostic and shared. It is published by the
`pipecat-dev-skills` marketplace and used by every repository whose changes feed
`pipecat-ai/docs`, so it must stay free of anything specific to one of them.

Everything repo-specific — what is in scope, how files map to pages, what a new
page looks like, where it gets registered — comes from that repo's **profile**,
`.claude/skills/update-docs/SOURCE_DOC_MAPPING.md`, which lives in the repo
being documented. The skill never hardcodes a source path, a page template, or a
navigation group. See `PROFILE_CONTRACT.md` for what a profile must provide.

## Arguments

```
/update-docs [DOCS_PATH]
```

- `DOCS_PATH` (optional): Path to the docs repository root. If not provided, ask the user.

Examples:

- `/update-docs /Users/me/src/docs`
- `/update-docs`

## Instructions

### Step 1: Resolve docs path and profile

If `DOCS_PATH` was provided as an argument, use it. Otherwise, ask the user for the path to their docs repository.

Verify the path exists and contains `docs.json`.

Read the profile at `.claude/skills/update-docs/SOURCE_DOC_MAPPING.md` in the
**source** repository. It defines this repo's scope, mapping tables, page
template, and registration steps. If it is missing, stop and say so — without it
there is no way to resolve a file to a page, and guessing produces edits to
pages that don't exist.

### Step 2: Create docs branch

Get the current source branch name:

```bash
git rev-parse --abbrev-ref HEAD
```

In the docs repo, create a new branch off main with a matching name:

```bash
cd DOCS_PATH && git checkout main && git pull && git checkout -b {branch-name}-docs
```

For example, if the source branch is `feat/new-service`, the docs branch becomes `feat/new-service-docs`.

All doc edits in subsequent steps are made on this branch.

### Step 3: Detect changed source files

Run:

```bash
git diff main..HEAD --name-only
```

Every source file under the roots the profile's **Scope** section names is in
scope. Repos ship public API well beyond their most obvious entry points, so
scope is defined as exclusions rather than an allowlist: a new directory is
covered the day it appears rather than when someone remembers to list it.

Exclude what the profile's Scope section excludes, plus build and cache
artifacts (`__pycache__/`, `*.pyc`, `node_modules/`, `dist/`) and re-export-only
index files that define nothing themselves.

Changes outside those roots — examples, CI config, the docs directory — don't
trigger doc updates on their own.

Then apply the profile's Skip list (Step 4), which names the small set of
genuinely internal files. Don't invent further reasons to drop a file here: being
a base class, a shared module, or "core architecture" is not one. Public
constructor parameters and observable behavior get documented wherever they live,
and a file whose page isn't obvious should reach Step 8 as a reported gap rather
than disappear.

### Step 4: Map source files to doc pages

For each changed source file, resolve the doc page to edit using the profile's
tables, in this order. **Confirm every candidate path exists in `DOCS_PATH`
before using it.**

1. **Skip list** — if the file matches the skip list, stop. It triggers no doc update.
2. **Base classes** — if the file is in the base-classes table, use every page it lists. A change here affects everything that inherits it, so check each page rather than stopping at the first.
3. **Non-standard locations** — if the file is in the non-standard table, use that entry as the candidate path and confirm it exists.
4. **Pattern match** — apply the pattern table to get a candidate path, then confirm the `.mdx` file exists (glob/`ls` it under `DOCS_PATH`). If it exists, use it.
5. **Search** — if no candidate from steps 2–4 exists on disk, grep `DOCS_PATH` for the file's main class or exported symbol name (see the profile's Search section).
6. **Unmapped** — if nothing resolves, treat the file as unmapped and report it in Step 8.

Never edit a path you haven't confirmed exists. If a candidate path doesn't resolve, fall through to the search step.

Reaching step 6 is a finding, not a dead end: an unmapped file means public API
with no home on the docs site, which is exactly what Step 8 exists to surface.
Never resolve a file by dropping it.

### Step 5: Analyze each source-doc pair

For each mapped pair:

1. **Read the full source file** to understand current state
2. **Read the diff** for that file: `git diff main..HEAD -- <source_file>`
3. **Read the current doc page** in full

Identify what changed by comparing source to docs. The profile's **Section
vocabulary** names the sections this repo's pages use and what each is built
from; check each one that applies:

- **Constructor / initializer parameters**: compare the signature to the Configuration section's `<ParamField>` entries
- **Options or settings objects**: compare the declared fields to the page's table for them
- **Event handlers**: compare registered events and handler signatures to the Event Handlers section
- **Class names / imports**: check that Usage examples reference correct names
- **Behavioral changes**: check whether the Notes section needs updating
- **Command-line surface**, where the repo ships one: compare flags, arguments, defaults, and exit behavior to the command reference

### Step 6: Make targeted edits

For each doc page that needs updates, edit **only the sections that need changes**. Preserve all other content exactly as-is.

#### Rules

- **Never remove content** unless the corresponding source code was removed
- **Never rewrite sections** that are already accurate
- **Match existing formatting** — if the page uses `<ParamField>` tags, use them; if it uses tables, use tables
- **Keep descriptions concise** — match the tone and length of surrounding content
- **Preserve CardGroup, links, and examples** unless they reference removed functionality
- **Don't touch frontmatter** unless the class was renamed

#### Proportionality

Match the size of the doc change to the size of the source change. A changed
default is a changed default: edit the value and, if it needs one, the clause
beside it. A two-line source change should not produce a paragraph. When a diff
suggests more prose than the change warrants, that prose is usually explaining
the change rather than the API.

Add a note only when the behavior would surprise **a reader who has never seen
the previous behavior**. Judge it with the diff covered up: if the note only
makes sense as an explanation of what changed, it belongs in the changelog, not
here.

A behavior change is not by itself a reason to add a note. The question is
whether the _new_ behavior needs explaining on its own terms. A default that
moved from `True` to `None` needs the default updated; it needs a note only if
`None` is confusing to someone meeting it for the first time — and then the note
explains `None`, not the move.

Do not add a `<Note>`, `<Warning>`, or `<Tip>` for a change that fits in the
sentence that is already there. Callouts are for behavior a reader would
otherwise get wrong, not for drawing attention to what a PR happened to touch.

#### Section-specific guidance

**Configuration** (constructor params):

- Use `<ParamField path="name" type="type" default="value">` format if the page already uses it
- Add new params in logical order (required first, then optional)
- Remove params that no longer exist in source
- Update types/defaults that changed

**Options / settings objects** (runtime settings):

- Use whichever form the page already uses — a markdown table or `<ParamField>` entries
- Match the field names and types from the declaring class
- Include the default values from the source

**Usage** (code examples):

- Update import paths, class names, and parameter names
- Only modify examples if they would break or be misleading with the new API
- Don't rewrite working examples just to add new optional params

**Notes**:

- Add notes for new behavioral gotchas or breaking changes
- Remove notes about limitations that were fixed
- Keep existing notes that are still accurate

**Event Handlers**:

- Update the event table and example code
- Add new events, remove deleted ones
- Update handler signatures if they changed

**Overview / Key Features / Prerequisites**:

- Only update if the PR fundamentally changes what the thing does (new capability, removed capability, renamed class)
- Most PRs will NOT need changes to these sections

#### Deprecations

When source marks something deprecated — a `DeprecationWarning`, a
`@deprecated` decorator, a docstring note, or a removal version — the doc entry
for it says so too. Mark the entry itself rather than adding a separate note
elsewhere on the page: readers find the parameter, not the changelog.

- Use the `deprecated` attribute on `<ParamField>` where the page uses ParamFields
- State the replacement and the removal version when source names them
- Never delete a deprecated item that still exists in source — a reader with it
  in their code needs to find it and learn what to move to

### Step 7: Update guides

Guides reference specific class names, parameters, imports, and code patterns.
After completing reference doc edits, check whether any guides need updates too.
The profile's **Guide directories** section lists the directories to search for
this repo.

For each changed source file, collect the class names, renamed parameters, and changed imports from the diff, then search those directories:

```bash
grep -rl "ClassName\|old_param_name" DOCS_PATH/<guide dirs from profile>
```

For each guide that references changed code:

1. Read the full guide
2. Update class names, parameter names, import paths, and code examples that are now incorrect
3. **Don't rewrite prose** — only fix the specific references that changed
4. Leave guides alone if they reference the subject generally but don't use any changed APIs

### Step 8: Identify doc gaps

After processing all mapped pairs, check for two kinds of gaps:

**Missing pages**: Source files that resolved to no doc page (pattern, non-standard table, and search all came up empty) and are not on the skip list. For each, report:

- The source file path
- The main class(es) or exported symbol(s) it defines
- Whether a new doc page should be created

**Missing sections**: Mapped doc pages missing a section the source implies —
a page with no Configuration section for a type that takes constructor
parameters, or no options table where the source declares a settings class.
Flag these and offer to add them.

If a new page is wanted, follow the profile's **New pages** section, which
provides the page template, the destination path, and every registration step
this repo requires (navigation, and any index or support-matrix page). Do all of
them: a page that exists but isn't registered is invisible.

#### Frontmatter conventions

A new page's `title` and `description` become its `llms.txt` entry and its
citation label in AI tools, and the docs repo's metadata lint enforces them:

- **title**: 50 chars max, no `- Pipecat` suffix (Mintlify appends it). Add a
  `sidebarTitle` when the title runs past 30 chars.
- **description**: 110-140 chars, naming the classes the page documents and the
  modality acronym (STT/TTS/LLM/VAD) where relevant. Must be unique site-wide,
  as must the effective unfurl title (`og:title` if set, else `title`) — add an
  `og:title` when another page already uses the same short title.

### Step 9: Format and regenerate llms.txt

The docs repo checks in `llms.txt` (a navigation-ordered index built from each
page's frontmatter) and `llms-full.txt` (every page's full body). Its metadata
lint fails when either is stale, so regenerate them after any page edit,
`docs.json` navigation change, or new page.

Prettier reflows MDX and `llms-full.txt` embeds the page bodies verbatim, so
formatting has to settle before generation:

```bash
cd DOCS_PATH
npx prettier --ignore-unknown --write <edited files>
node scripts/gen-llms-txt.mjs
```

Commit the doc edits together with the regenerated `llms.txt` and
`llms-full.txt`. Generating before formatting leaves them stale — as does
relying on the repo's pre-commit hook, which formats pages after generation has
already run.

`node scripts/docs-meta-lint.mjs` reports the same staleness and frontmatter
findings CI will.

### Step 10: Output summary

After all edits are complete, print a summary:

```
## Documentation Updates

### Updated reference pages
- `<page path>` — what changed, and in which section

### Updated guides
- `<guide path>` — what changed

### New pages
- `<page path>` — created, plus every place it was registered

### Unmapped source files
- `<source path>` — ClassName (no doc page exists)

### Skipped files
- `<source path>` — why
```

## Guidelines

- **Write for a future reader, not the diff** — docs describe the API as it currently stands. Never narrate the change itself: no "newly added," "this replaces," "recently changed," or references to prior behavior. A reader landing on the page should see no sign that a PR just edited it. Match the weight of the prose to the feature — a routine new parameter gets a one-line description, not a paragraph.
- **Don't carry the changelog's reasoning into the docs** — the PR body and the changelog entry argue for a change to someone who knew the old behavior. The docs describe the current state to someone who doesn't. Those need different prose, so the changelog is a source of _facts_ here — the new default, the new name, what a parameter now does — and never a source of sentences. Copying its justification across is the most common way a small change turns into a paragraph, and it survives the rule above because a justification carries no "newly" or "previously" to strip out.
- **Avoid LLM tells** — write plainly. Skip filler and AI-signalling phrases ("delve," "seamless," "leverage," "it is worth noting," "this underscores"), formulaic "not just X, but Y" contrasts, and overuse of em dashes or boldface. Never leave placeholder text (`[X]`, `{placeholder}`) or assistant meta ("I hope this helps") in a page — this skill runs unattended in CI, so nothing downstream will catch it.
- **Keep code and prose in sync** — when a page names a parameter, class, or identifier, spell it in prose exactly as the source and the `<ParamField>`/table entry do. After editing a code example or renaming a param, re-read the surrounding prose for stale references.
- **Backtick inline technical terms** — wrap parameter names, class names, filenames, env vars, and config keys in backticks when they appear in prose (Overview, Notes, descriptions). Structured elements like `<ParamField>` already format these inside tables.
- **Be conservative** — only change what the diff warrants. Don't "improve" docs beyond what changed in source.
- **Read before editing** — always read the full doc page before making changes so you understand the existing structure.
- **Preserve voice** — match the writing style of the existing doc page, don't impose a different tone.
- **One PR at a time** — this skill operates on the current branch's diff against main. Don't look at other branches.
- **Parallel analysis** — when multiple source files map to different doc pages, analyze and edit them in parallel for efficiency.
- **Shared source files** — base classes and shared modules affect everything that imports them. Check which pages cover those consumers and update all of them.

## Checklist

Before finishing, verify:

- [ ] The source repo's profile was read, and every changed file was checked against its tables
- [ ] Each doc page edit matches the actual source code change (not guessed)
- [ ] No content was removed unless the corresponding source was removed
- [ ] New parameters have accurate types and defaults from source
- [ ] Deprecated items are marked as deprecated rather than deleted or left unmarked
- [ ] The edit is no larger than the change: no note or paragraph that only makes sense as an explanation of what changed
- [ ] Formatting matches the existing page style
- [ ] Guides referencing changed APIs were checked and updated
- [ ] New pages were registered everywhere the profile requires
- [ ] Edited pages were formatted, then `llms.txt` and `llms-full.txt` regenerated and committed
- [ ] Unmapped files were reported
