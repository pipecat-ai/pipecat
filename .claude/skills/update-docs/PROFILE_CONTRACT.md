# update-docs — writing a profile for a repo

`SKILL.md` in this directory is the canonical, repo-agnostic instruction set for
the `update-docs` automation. It is published by the `pipecat-dev-skills`
marketplace and shared by every repository whose changes feed `pipecat-ai/docs`.

It lives in one place because it previously did not. Copies in two repos drifted
to 390 and 117 lines, the smaller missing every rule added after it was copied.
With four more repos to onboard, per-repo copies would mean six places to fix
each future change.

## What each repo provides

The skill supplies the workflow. Each documented repo supplies a **profile** at
`.claude/skills/update-docs/SOURCE_DOC_MAPPING.md` — everything the skill looks
up but cannot know.

```
consuming repo (e.g. pipecat-cloud)        pipecat
├── .github/workflows/update-docs.yml      └── .claude/skills/update-docs/
└── .claude/skills/update-docs/                ├── SKILL.md              ← shared
    └── SOURCE_DOC_MAPPING.md                  ├── PROFILE_CONTRACT.md   ← this file
        ↑ repo-specific                        └── SOURCE_DOC_MAPPING.md ← pipecat's own profile
```

Locally, installing the plugin makes `/update-docs` work in any repo that has a
profile. In CI, a workflow that does not already have this repo checked out
fetches just the skill:

```yaml
- uses: actions/checkout@v4
  with:
    repository: pipecat-ai/pipecat
    sparse-checkout: .claude/skills/update-docs
    path: _skill
    fetch-depth: 1
```

## Required sections

`SKILL.md` reads these by name. A profile missing one leaves the corresponding
step with nothing to apply, so write all of them.

| Section | What it defines | Used by |
| --- | --- | --- |
| **Scope** | Source roots in scope, and what to exclude within them. State exclusions, not an allowlist, so new directories are covered on the day they appear. | Step 3 |
| **Skip list** | The few genuinely internal files that trigger no doc update. Being a base class or "core architecture" does not qualify. | Step 4.1 |
| **Base classes** | Files whose changes affect many pages, each mapped to *every* page to check. | Step 4.2 |
| **Non-standard locations** | Files whose page can't be derived by pattern. | Step 4.3 |
| **Patterns** | Source path → doc path rules covering the bulk of the repo. | Step 4.4 |
| **Search** | What symbol to grep for when the tables come up empty. | Step 4.5 |
| **Section vocabulary** | The sections this repo's pages use, and what each is built from. | Step 5 |
| **Guide directories** | Doc directories holding prose that cites this repo's API. | Step 7 |
| **New pages** | Page template, destination path, and *every* registration step — navigation plus any index or support-matrix page. | Step 8 |

## Writing one

Start from the profile of whichever repo is closest in shape, then work through
the table above. Two things are worth doing before trusting it:

1. **Resolve backwards.** For a sample of doc pages, ask which source file the
   profile would map to them. A page no rule reaches is a page the automation
   will never update.
2. **Run it on a merged PR.** `workflow_dispatch` accepts a PR number, so a
   known-good change from last month is a free test with a reviewable diff.

The test for the Skip list is not "is this internal architecture" but **"can
someone change or observe this without subclassing it?"** If yes, it has a page
somewhere and belongs in a mapping table.

## Changing the shared skill

An edit to `SKILL.md` changes behavior for every consuming repo at once — that
is the point, and the risk. Prefer changes that make a rule clearer over ones
that add a rule, and when guidance is only needed by one repo, put it in that
repo's profile instead.

`SKILL.md` also encodes conventions owned by `pipecat-ai/docs` — the `llms.txt`
regeneration ordering, the frontmatter length bands, `docs.json` structure. When
those change there, this file has to follow.
