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

Verify the path exists and contains `api-reference/server/services/` subdirectory.

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
- `src/pipecat/flows/**/*.py` (Pipecat Flows)

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

Guides at `DOCS_PATH/pipecat/` and `DOCS_PATH/pipecat-flows/` reference specific class names, parameters, imports, and code patterns. After completing reference doc edits, check if any guides need updates too.

For each changed source file, collect the class names, renamed parameters, and changed imports from the diff. Search the guides directory:

```bash
grep -rl "ClassName\|old_param_name" DOCS_PATH/pipecat/ DOCS_PATH/pipecat-flows/
```

For each guide that references changed code:

1. Read the full guide
2. Update class names, parameter names, import paths, and code examples that are now incorrect
3. **Don't rewrite prose** — only fix the specific references that changed
4. Leave guides alone if they reference the service generally but don't use any changed APIs

Guide directories:

- `pipecat/learn/` — conceptual tutorials (pipeline, LLM, STT, TTS, etc.)
- `pipecat/fundamentals/` — practical how-tos (metrics, recording, transcripts, etc.)
- `pipecat/features/` — feature-specific guides (Gemini Live, OpenAI audio, WhatsApp, etc.)
- `pipecat/telephony/` — telephony integration guides (Twilio, Plivo, Telnyx, etc.)
- `pipecat-flows/guides/` — Pipecat Flows guides (nodes-and-messages, functions, context-strategies, state-management, actions); check these when `src/pipecat/flows/**` changed

### Step 8: Identify doc gaps

After processing all mapped pairs, check for two kinds of gaps:

**Missing pages**: Source files that had no doc page mapping (neither tier 1, 2, nor 3) and are not marked as "(skip)". For each, tell the user:

- The source file path
- The main class(es) it defines
- Whether a new doc page should be created

**Missing sections**: Mapped doc pages that are missing standard sections compared to the source. For example, a transport page with no Configuration section, or a service page with no InputParams table when the source defines `InputParams(BaseModel)`. Flag these and offer to add the missing sections.

If the user wants a new page, do all three of the following:

#### 8a: Create the doc page

Create the new `.mdx` file under `DOCS_PATH/api-reference/server/services/{category}/{provider}.mdx` using this template structure:

````
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
uv add "pipecat-ai[package-name]"
````

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

````

#### 8b: Add to docs.json

Add the new page path to `DOCS_PATH/docs.json` in the correct navigation group. The path format is `api-reference/server/services/{category}/{provider}` (without the `.mdx` extension).

Find the matching group in the navigation structure:
- **STT** → `"group": "Speech-to-Text"` under Services
- **TTS** → `"group": "Text-to-Speech"` under Services
- **LLM** → `"group": "LLM"` under Services
- **S2S** → `"group": "Speech-to-Speech"` under Services
- **Transport** → `"group": "Transport"` under Services
- **Serializer** → `"group": "Serializers"` under Services
- **Image generation** → `"group": "Image Generation"` under Services
- **Video** → `"group": "Video"` under Services
- **Memory** → `"group": "Memory"` under Services
- **Vision** → `"group": "Vision"` under Services
- **Analytics** → `"group": "Analytics & Monitoring"` under Services

Insert the new entry **alphabetically** within the group's `pages` array. For example, adding a new STT service "foo":
```json
{
  "group": "Speech-to-Text",
  "pages": [
    "api-reference/server/services/stt/assemblyai",
    "api-reference/server/services/stt/aws",
    ...
    "api-reference/server/services/stt/foo",
    ...
  ]
}
````

#### 8c: Add to supported-services.mdx

Add a new row to the correct category table in `DOCS_PATH/api-reference/server/services/supported-services.mdx`.

Use this format:

```
| [DisplayName](/api-reference/server/services/{category}/{provider}) | `uv add "pipecat-ai[package]"` |
```

To determine the correct values:

- **DisplayName**: Use the service's human-readable name (e.g., "ElevenLabs", "AWS Polly", "Google Gemini")
- **package**: Look at the service's `pyproject.toml` extras or the import pattern in the source code. For example, if the service is in `src/pipecat/services/foo/`, the package is typically `foo`.
- If no pip dependencies are required, use `No dependencies required` instead.

Insert the new row **alphabetically** within the table. Match the column alignment of the existing rows.

### Step 9: Output summary

After all edits are complete, print a summary:

```
## Documentation Updates

### Updated reference pages
- `api-reference/server/services/stt/deepgram.mdx` — Updated Configuration (added `new_param`), InputParams (updated `language` default)
- `api-reference/server/services/tts/elevenlabs.mdx` — Updated Event Handlers (added `on_connected`)
- `api-reference/pipecat-flows/flow-manager.mdx` — Updated FlowManager constructor (added `new_param`)

### Updated guides
- `pipecat/learn/speech-to-text.mdx` — Updated code example (renamed `old_param` → `new_param`)
- `pipecat-flows/guides/state-management.mdx` — Updated FlowManager init example

### New service pages
- `api-reference/server/services/tts/newprovider.mdx` — Created page, added to docs.json (Text-to-Speech), added to supported-services.mdx

### Unmapped source files
- `src/pipecat/services/newprovider/tts.py` — NewProviderTTSService (no doc page exists)

### Skipped files
- `src/pipecat/services/ai_service.py` — internal base class
```

## Guidelines

- **Write for a future reader, not the diff** — docs describe the API as it currently stands. Never narrate the change itself: no "newly added," "this replaces," "recently changed," or references to prior behavior. A reader landing on the page should see no sign that a PR just edited it. Match the weight of the prose to the feature — a routine new parameter gets a one-line description, not a paragraph.
- **Avoid LLM tells** — write plainly. Skip filler and AI-signalling phrases ("delve," "seamless," "leverage," "it is worth noting," "this underscores"), formulaic "not just X, but Y" contrasts, and overuse of em dashes or boldface. Never leave placeholder text (`[X]`, `{placeholder}`) or assistant meta ("I hope this helps") in a page — this skill runs unattended in CI, so nothing downstream will catch it.
- **Keep code and prose in sync** — when a page names a parameter, class, or identifier, spell it in prose exactly as the source and the `<ParamField>`/table entry do. After editing a code example or renaming a param, re-read the surrounding prose for stale references.
- **Backtick inline technical terms** — wrap parameter names, class names, filenames, env vars, and config keys in backticks when they appear in prose (Overview, Notes, descriptions). Structured elements like `<ParamField>` already format these inside tables.
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
- [ ] New service pages were added to `docs.json` in the correct group, alphabetically
- [ ] New service pages were added to `supported-services.mdx` in the correct table, alphabetically
- [ ] Unmapped files were reported to the user
