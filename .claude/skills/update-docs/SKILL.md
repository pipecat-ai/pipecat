---
name: update-docs
description: Update documentation pages to match source code changes on the current branch
---

Update documentation pages to reflect source code changes on the current branch. Analyzes the diff against main, maps changed source files to their corresponding doc pages, and makes targeted edits.

## Arguments

```
/update-docs [DOCS_PATH]
```

- `DOCS_PATH` (optional): Path to the docs repository root. If not provided, ask the user.

Examples:
- `/update-docs /Users/me/src/docs`
- `/update-docs`

## Instructions

### Step 1: Resolve docs path

If `DOCS_PATH` was provided as an argument, use it. Otherwise, ask the user for the path to their docs repository.

Verify the path exists and contains `server/services/` subdirectory.

### Step 2: Create docs branch

Get the current pipecat branch name:
```bash
git rev-parse --abbrev-ref HEAD
```

In the docs repo, create a new branch off main with a matching name:
```bash
cd DOCS_PATH && git checkout main && git pull && git checkout -b {branch-name}-docs
```

For example, if the pipecat branch is `feat/new-service`, the docs branch becomes `feat/new-service-docs`.

All doc edits in subsequent steps are made on this branch.

### Step 3: Detect changed source files

Run:
```bash
git diff main..HEAD --name-only
```

Filter to files that could affect documentation:
- `src/pipecat/services/**/*.py` (service implementations)
- `src/pipecat/transports/**/*.py` (transport implementations)
- `src/pipecat/serializers/**/*.py` (serializer implementations)
- `src/pipecat/processors/**/*.py` (processor implementations)
- `src/pipecat/audio/**/*.py` (audio utilities)
- `src/pipecat/turns/**/*.py` (turn management)
- `src/pipecat/observers/**/*.py` (observers)
- `src/pipecat/pipeline/**/*.py` (pipeline core)

Ignore `__init__.py`, `__pycache__`, test files, and files that only contain type re-exports.

### Step 4: Map source files to doc pages

For each changed source file, find the corresponding doc page. Read the mapping file at `.claude/skills/update-docs/SOURCE_DOC_MAPPING.md` and apply its tiered lookup: tier 1 (known exceptions) → tier 2 (pattern matching) → tier 3 (search fallback). **First match wins.**

### Step 5: Analyze each source-doc pair

For each mapped pair:

1. **Read the full source file** to understand current state
2. **Read the diff** for that file: `git diff main..HEAD -- <source_file>`
3. **Read the current doc page** in full

Identify what changed by comparing source to docs:

- **Constructor parameters**: Compare `__init__` signature to the Configuration section's `<ParamField>` entries
- **InputParams fields**: Compare `InputParams(BaseModel)` class fields to the InputParams table
- **Event handlers**: Compare `_register_event_handler` calls and event handler definitions to Event Handlers section
- **Class names / imports**: Check if Usage examples reference correct names
- **Behavioral changes**: Check if Notes section needs updating

### Step 6: Make targeted edits

For each doc page that needs updates, edit **only the sections that need changes**. Preserve all other content exactly as-is.

#### Rules

- **Never remove content** unless the corresponding source code was removed
- **Never rewrite sections** that are already accurate
- **Match existing formatting** — if the page uses `<ParamField>` tags, use them; if it uses tables, use tables
- **Keep descriptions concise** — match the tone and length of surrounding content
- **Preserve CardGroup, links, and examples** unless they reference removed functionality
- **Don't touch frontmatter** unless the class was renamed

#### Section-specific guidance

**Configuration** (constructor params):
- Use `<ParamField path="name" type="type" default="value">` format if the page already uses it
- Add new params in logical order (required first, then optional)
- Remove params that no longer exist in source
- Update types/defaults that changed

**InputParams** (runtime settings):
- Use markdown table format: `| Parameter | Type | Default | Description |`
- Match the field names and types from the `InputParams(BaseModel)` class
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
- Only update if the PR fundamentally changes what the service does (new capability, removed capability, renamed class)
- Most PRs will NOT need changes to these sections

### Step 7: Update guides

Guides at `DOCS_PATH/guides/` reference specific class names, parameters, imports, and code patterns. After completing reference doc edits, check if any guides need updates too.

For each changed source file, collect the class names, renamed parameters, and changed imports from the diff. Search the guides directory:
```bash
grep -rl "ClassName\|old_param_name" DOCS_PATH/guides/
```

For each guide that references changed code:
1. Read the full guide
2. Update class names, parameter names, import paths, and code examples that are now incorrect
3. **Don't rewrite prose** — only fix the specific references that changed
4. Leave guides alone if they reference the service generally but don't use any changed APIs

Guide directories:
- `guides/learn/` — conceptual tutorials (pipeline, LLM, STT, TTS, etc.)
- `guides/fundamentals/` — practical how-tos (metrics, recording, transcripts, etc.)
- `guides/features/` — feature-specific guides (Gemini Live, OpenAI audio, WhatsApp, etc.)
- `guides/telephony/` — telephony integration guides (Twilio, Plivo, Telnyx, etc.)

### Step 8: Identify doc gaps

After processing all mapped pairs, check for two kinds of gaps:

**Missing pages**: Source files that had no doc page mapping (neither tier 1, 2, nor 3) and are not marked as "(skip)". For each, tell the user:
- The source file path
- The main class(es) it defines
- Whether a new doc page should be created

**Missing sections**: Mapped doc pages that are missing standard sections compared to the source. For example, a transport page with no Configuration section, or a service page with no InputParams table when the source defines `InputParams(BaseModel)`. Flag these and offer to add the missing sections.

If the user wants a new page, create it using this template structure:
```
---
title: "Service Name"
description: "Brief description"
---

## Overview

[Description from class docstring or source analysis]

<CardGroup cols={2}>
  [Cards for API reference and examples if available]
</CardGroup>

## Installation

```bash
pip install "pipecat-ai[package-name]"
```

## Prerequisites

[Environment variables and account setup]

## Configuration

[ParamField entries for constructor params]

## InputParams

[Table of InputParams fields, if the service has them]

## Usage

### Basic Setup

```python
[Minimal working example]
```

## Notes

[Important caveats]

## Event Handlers

[Event table and example code]
```

### Step 9: Output summary

After all edits are complete, print a summary:

```
## Documentation Updates

### Updated reference pages
- `server/services/stt/deepgram.mdx` — Updated Configuration (added `new_param`), InputParams (updated `language` default)
- `server/services/tts/elevenlabs.mdx` — Updated Event Handlers (added `on_connected`)

### Updated guides
- `guides/learn/speech-to-text.mdx` — Updated code example (renamed `old_param` → `new_param`)

### Unmapped source files
- `src/pipecat/services/newprovider/tts.py` — NewProviderTTSService (no doc page exists)

### Skipped files
- `src/pipecat/services/ai_service.py` — internal base class
```

## Guidelines

- **Be conservative** — only change what the diff warrants. Don't "improve" docs beyond what changed in source.
- **Read before editing** — always read the full doc page before making changes so you understand the existing structure.
- **Preserve voice** — match the writing style of the existing doc page, don't impose a different tone.
- **One PR at a time** — this skill operates on the current branch's diff against main. Don't look at other branches.
- **Parallel analysis** — when multiple source files map to different doc pages, analyze and edit them in parallel for efficiency.
- **Shared source files** — files like `services/google/google.py` are shared bases. Check which services import from them and update all affected doc pages.

## Checklist

Before finishing, verify:

- [ ] All changed source files were checked against the mapping table
- [ ] Each doc page edit matches the actual source code change (not guessed)
- [ ] No content was removed unless the corresponding source was removed
- [ ] New parameters have accurate types and defaults from source
- [ ] Formatting matches the existing page style
- [ ] Guides referencing changed APIs were checked and updated
- [ ] Unmapped files were reported to the user
