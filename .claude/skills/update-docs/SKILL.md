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

How pages read — what to document, what to leave out, page templates, prose —
comes from the docs repo's style guide, `DOCS_PATH/STYLE.md`. This skill covers
the process: finding what changed, deciding what the docs need, and checking
the result.

`REVIEW.md` is a second pass over a finished update, run in a fresh session
that did not write it. The automation runs it after this skill; run it by hand
the same way after a local update.

## Arguments

```
/update-docs [DOCS_PATH]
```

- `DOCS_PATH` (optional): Path to the docs repository root. If not provided, ask the user.

Examples:

- `/update-docs /Users/me/src/docs`
- `/update-docs`

## Instructions

### Step 1: Resolve docs path, profile, and style guide

If `DOCS_PATH` was provided as an argument, use it. Otherwise, ask the user for the path to their docs repository.

Verify the path exists and contains `docs.json`.

Read the profile at `.claude/skills/update-docs/SOURCE_DOC_MAPPING.md` in the
**source** repository. It defines this repo's scope, mapping tables, page
template, and registration steps. If it is missing, stop and say so — without it
there is no way to resolve a file to a page, and guessing produces edits to
pages that don't exist.

Read `DOCS_PATH/STYLE.md` in full. Every edit follows it. Where it and this
skill disagree about how a page reads, the style guide wins. If it is missing,
follow the Guidelines at the end of this skill.

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

### Step 5: Work out what the docs need

#### List what changed for a reader

Before opening any page for editing, read the full PR: its description
(including any **Breaking Changes** section), every changelog fragment it adds,
and the diff. Then write down, for yourself, the list of **reader-facing
changes**: things a user can now set, call, observe, or act on differently.
Every bullet of a **Breaking Changes** section goes on the list, one item per
bullet, unless it only affects code that subclasses or calls internals; it is
there because a user's code or expectations change.

For each item, record the facts from the source, not from prose about it:

- The name, type, and default as declared in the code
- Allowed values, limits, and required combinations
- What raises, and when
- Side effects a caller would notice: reconnects, clipping, values passed
  through to the provider, ordering
- What it replaces, if anything is deprecated or removed

Verify every consequence you intend to state by reading the code that produces
it. The PR description, changelog, and docstrings say what the author intended;
the code says what happens. When they disagree, document the code, and check
the Breaking Changes section for the cases the summary leaves out.

Docstrings, comments, PR descriptions, and changelog entries are sources of
**facts**, never of sentences. They are written for maintainers and carry
rationale ("Early Access", error codes, why the code works this way) that a
reader does not need. Restate the fact for the reader in the page's own words.

**Before deciding there is nothing to document, check what the docs already
say.** Search the mapped pages for the behavior the diff touches, and compare
each statement with the code after the change. A statement the change made
false is an item on the list, even when the code change is only a docstring or
comment: docs often repeat the old docstring's wording, and it is now wrong.

**If the list is still empty, the change needs no docs.** Typical cases: an
internal refactor, a fix that restores documented behavior, a change to log
text. Record the reason (the automation's NOOP line) and stop. Do not write
something because a file changed.

#### Analyze each source-doc pair

For each mapped pair:

1. **Read the full source file** to understand current state
2. **Read the diff** for that file: `git diff main..HEAD -- <source_file>`
3. **Read the current doc page** in full

For each item on the Step 5 list, find where the page already covers it, if
anywhere. The profile's **Section vocabulary** names the sections this repo's
pages use and what each is built from; check each one that applies:

- **Constructor / initializer parameters**: compare the signature to the Configuration section's `<ParamField>` entries
- **Options or settings objects**: compare the declared fields to the page's table for them
- **Event handlers**: compare registered events and handler signatures to the Event Handlers section
- **Class names / imports**: check that Usage examples reference correct names
- **Behavioral changes**: check whether any statement on the page is now false
- **Command-line surface**, where the repo ships one: compare flags, arguments, defaults, and exit behavior to the command reference

### Step 6: Make targeted edits

Each item on the Step 5 list goes in the one place a reader would look for it,
in the page's existing format. STYLE.md's "Changing Existing Pages" table gives
the expected size of each kind of edit: a new parameter is one entry, with no
Note, no example, and no overview change.

#### Rules

- **Edit a line only if the change makes it false or incomplete.** Do not
  reword, tighten, or merge neighboring sentences or bullets. Rewording text the
  change didn't touch is where errors and lost facts come from.
- **A line that contradicts or repeats an entry you added counts as made
  false.** After adding or changing an entry, read the rest of the page for
  statements about the same thing. Delete one that only repeats the entry; fix
  one that contradicts it (checking the code to see which is right).
- **State each fact once.** Before adding a sentence, search the page for the
  fact. If it is already there, update that occurrence instead of adding
  another. A Note must not restate a `<ParamField>` or table row.
- **Fix what the change made wrong**, including examples and recipes. An example
  that now misbehaves, uses a deprecated parameter, or contradicts the new
  behavior is updated even though it was accurate before.
- **Never remove content** unless the corresponding source was removed, or the
  change made it false.
- **Match existing formatting** — if the page uses `<ParamField>` tags, use them; if it uses tables, use tables.
- **Preserve CardGroup, links, and examples** unless they reference removed functionality.
- **Don't touch frontmatter** unless the class was renamed.
- **Undocumented surface the change touches** — when the change modifies
  public API that was never documented, document it, briefly: the entries
  themselves, without a new section of explanation.

#### Deprecations

When source marks something deprecated — a `DeprecationWarning`, a
`@deprecated` decorator, a docstring note, or a removal version — the doc entry
for it says so too, in the format STYLE.md gives. Mark the entry itself rather
than adding a separate note elsewhere on the page.

- State the replacement and the removal version when source names them
- Never delete a deprecated item that still exists in source
- When the deprecated API has no doc entry, but a page about it has a migration
  table or section, add a row there. Otherwise report it in Step 8.

### Step 7: Find what else the change made wrong

Pages other than the mapped ones can be made wrong by the change. Search
`DOCS_PATH` (the profile's **Guide directories**, plus `api-reference/`) for
each of these:

1. **Changed names**: class names, renamed parameters, and changed imports from
   the diff. Update every reference that is now incorrect.
2. **Siblings of a new class.** When the change adds a class to a family
   (a new classifier, service, transport, serializer), find the pages that list
   its siblings: grep for two or three sibling class names together, and check
   supported-services and comparison tables, feature lists, and counts ("two
   classifiers"). Add the new class to every one, in that list's own form: a
   comparison table gains a column or row, not a sentence beside it, and a count
   changes along with the table or list it counts.
3. **Recipes that use changed behavior.** When behavior changes, grep for the
   affected class, event, and parameters across all of `DOCS_PATH`. For each
   snippet or instruction you find, trace what it does under the new behavior,
   step by step: what runs, what fires, what the user hears. If the result is
   now wrong, duplicated, or redundant (for example, the snippet does by hand
   what the framework now does by default), fix the snippet.
4. **Migration tables** that should gain a row for a deprecation.

For each guide that references changed code: read it in full, fix the specific
references that changed, and don't rewrite its prose. Leave guides alone if
they mention the subject but don't use any changed API.

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
them: a page that exists but isn't registered is invisible. Follow STYLE.md for
the page's structure, and CLAUDE.md in the docs repo for frontmatter.

#### Frontmatter conventions

A new page's `title` and `description` become its `llms.txt` entry and its
citation label in AI tools, and the docs repo's metadata lint enforces them:

- **title**: 50 chars max, no `- Pipecat` suffix (Mintlify appends it). Add a
  `sidebarTitle` when the title runs past 30 chars.
- **description**: 110-140 chars, naming the classes the page documents and the
  modality acronym (STT/TTS/LLM/VAD) where relevant. Must be unique site-wide,
  as must the effective unfurl title (`og:title` if set, else `title`) — add an
  `og:title` when another page already uses the same short title.

#### Review every edited page

Last, re-read each page you edited, in full, as a reader who never saw the diff:

- Every item on the Step 5 list appears, once, in the right place
- No fact appears twice on the page (check the overview, feature lists, Notes,
  and Usage comments against the entries)
- No sentence narrates the change or explains maintainer rationale
- Every example still runs against the current API: no deprecated parameters,
  no undefined names
- Counts and lists in prose match the tables they describe
- The diff contains only the edits you intended: `git diff` in `DOCS_PATH`

Then run each command in the profile's **Checks** section, if it has one. Fix
every finding on a page you edited, whatever caused it: the docs repo's CI runs
the same checks on every page a PR changes and fails the PR on any finding, so
this overrides Step 6's rule against editing lines the change didn't make
false. Fix each in the page's format and STYLE.md's (types in Python syntax).
A finding on a page you didn't edit is yours to fix when this change caused it
(a setting the diff added, a default it changed); otherwise leave it and list
it under gaps.

Fix anything that fails before continuing.

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

STYLE.md is the authority on how pages read. These are the minimum when it is
absent:

- **Write for a future reader, not the diff** — docs describe the API as it currently stands. Never narrate the change itself: no "newly added," "this replaces," "recently changed," or references to prior behavior.
- **No rationale, no internals** — leave out why the code works this way and how it works inside, unless the reader must account for it in their own code.
- **Callouts only for what a reader would otherwise get wrong** — a fact that fits in an existing sentence goes in that sentence.
- **Keep code and prose in sync** — when a page names a parameter, class, or identifier, spell it in prose exactly as the source and the `<ParamField>`/table entry do.
- **Backtick inline technical terms** — parameter names, class names, filenames, env vars, and config keys.
- **Plain prose** — no filler, no placeholder text, no assistant meta. This skill runs unattended in CI, so nothing downstream will catch it.

Process:

- **Read before editing** — always read the full doc page before making changes so you understand the existing structure.
- **One PR at a time** — this skill operates on the current branch's diff against main. Don't look at other branches.
- **Shared source files** — base classes and shared modules affect everything that imports them. Check which pages cover those consumers and update all of them.

## Checklist

Before finishing, verify:

- [ ] The profile and STYLE.md were read
- [ ] The Step 5 list of reader-facing changes was written, with facts checked against the code; an empty list ended the run as a NOOP
- [ ] Each item on the list is documented once, in the page's existing format
- [ ] No line was edited that the change didn't make false or incomplete
- [ ] Deprecated items are marked, and migration tables gained rows
- [ ] Sibling lists, recipes, and guides affected by the change were searched (Step 7)
- [ ] Every edited page passed the review at the end of Step 8, including the profile's Checks
- [ ] New pages were registered everywhere the profile requires
- [ ] Edited pages were formatted, then `llms.txt` and `llms-full.txt` regenerated and committed
- [ ] Unmapped files were reported
