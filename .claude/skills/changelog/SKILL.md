---
name: changelog
description: Create changelog files for important commits in a PR
---

Create changelog files for the important commits in this PR. The PR number is provided as an argument.

## Instructions

1. Skip changelog for: documentation-only, internal refactoring, test-only, CI changes.

2. First, check what commits are on the current branch compared to main:
   ```
   git log main..HEAD --oneline
   ```

3. For each significant change, create a changelog file in the `changelog/` folder using the format:
   Allowed types: `added`, `changed`, `deprecated`, `removed`, `fixed`, `security`, `performance`, `other`
   - `{PR_NUMBER}.added.md` - for new features
   - `{PR_NUMBER}.added.2.md`, `{PR_NUMBER}.added.3.md` - for additional entries of the same type
   - `{PR_NUMBER}.changed.md` - for changes to existing functionality
   - `{PR_NUMBER}.fixed.md` - for bug fixes
   - `{PR_NUMBER}.deprecated.md` - for deprecations
   - `{PR_NUMBER}.removed.md` - for removed features
   - `{PR_NUMBER}.security.md` - for security fixes
   - `{PR_NUMBER}.performance.md` - for performance improvements
   - `{PR_NUMBER}.other.md` - for other changes

4. Each changelog file should at least contain a main single line starting with `- ` followed by a clear description of the change. No line wrapping.

5. If the change is complicated, changelog files can have indented lines after the main line with additional details or code samples.

6. Use ⚠️ emoji prefix for breaking changes.

7. **Write changes in user-facing terms first.** Lead with what users of the framework will notice: new APIs, changed behavior, new parameters, fixed bugs they might have hit, etc. Implementation details (internal refactoring, how something is wired up under the hood) can be included as secondary context after the user-facing description, but should never be the *only* content of a changelog entry when there is a user-visible effect.

   **Good** (user-facing first, implementation detail as context):
   ```
   - Turn completion instructions now persist correctly across full context updates when using `system_instruction`. Previously they were injected as a context system message, which caused warning spam and didn't survive context updates.
   ```

   **Bad** (implementation detail only, no user-facing framing):
   ```
   - Fixed turn completion instructions being injected as a context system message instead of using `system_instruction`.
   ```

   Ask yourself: "If I'm a developer building on Pipecat, what would I notice changed?" Start there.

8. **For `deprecated` and `removed` entries, follow the phrasing patterns in "Deprecation and removal phrasing" below.** These entries are machine-read by `check_deprecation` (Pipecat Context Hub) so agents can warn users off stale APIs; ambiguous phrasing makes it mis-key entries — most dangerously, flagging a *current* API as deprecated when that API is only named as the replacement.

9. **For each `deprecated`/`removed` entry, also add a record to `deprecations.json`** (repo root) as described in "The deprecations.json registry" below. The prose `.md` is for humans; `deprecations.json` is the structured form tools consume directly, so they never have to parse prose. You already know every field while writing the entry — add both in the same PR.

## Example

For PR #3519 with a new feature and a bug fix:

`changelog/3519.added.md`:
```
- Added `SomeNewFeature` for doing something useful.
```

`changelog/3519.fixed.md`:
```
- Fixed an issue where something was not working correctly in some user-visible scenario. The root cause was an internal implementation detail.
```

## Deprecation and removal phrasing

`deprecated` and `removed` entries are read by `check_deprecation` to tell an
agent whether an API it's about to use is stale and what to use instead. The
*phrasing* decides whether it answers correctly. The failure mode that matters:
when a current API is merely named as the replacement (or as the class that owns
a deprecated member), loose phrasing makes the tool report that **current** API
as deprecated, sending agents after a fix that doesn't exist.

**Four rules make any entry parseable:**

1. The deprecated/removed symbol is the **first** backticked token in the line.
2. The replacement (if any) comes **after an explicit verb** — "renamed to",
   "moved to", "split into", "use", "now part of".
3. Members are written **dotted**: `` `ClassName.member` `` — never
   `` `ClassName`: `member` ``, `` `ClassName`'s `member` ``, or
   `` `ClassName` `member` parameter ``.
4. When there is **no replacement**, say so explicitly: end with
   "No replacement."

**Templates by kind** (`X`, `Y` are backticked symbols; name the *old* symbol
first, the replacement after the verb — never the reverse):

| Situation | Write it as |
|---|---|
| Renamed (same thing, new name) | `` `X` has been renamed to `Y`. `` |
| Moved (import path changed) | `` `pkg.x` has moved to `pkg.y`. `` |
| Split into several | `` `pkg.x` has been split into `pkg.a` and `pkg.b`. `` |
| A member / param / field deprecated | `` `X.member` is deprecated; use `X.other` instead. `` |
| Folded into an existing API | `` `X` has been removed; its functionality is now part of `Y`. `` |
| Replaced by a *different* existing API | `` `X` has been removed; use the existing `Y` instead. `` |
| Behavior changed, option gone | `` `X` has been removed; <behavior> is now the default. No replacement. `` |
| Removed, nothing replaces it | `` `X` has been removed. No replacement. `` |
| Removed because a provider / dependency went away | `` `X` has been removed (provider discontinued). No replacement. `` |

**Good** (deprecated member — the class stays current):
```
- `TTSService.text_aggregator` is deprecated; pass a `TextAggregator` to the constructor instead.
```
**Bad** (reads as the whole class being deprecated → flags a current API):
```
- `TTSService`: `text_aggregator` init param deprecated.
```

**Good** (rename — old name first, new name after the verb):
```
- ⚠️ `PipelineTask` has been renamed to `PipelineWorker`. The old name still resolves but emits a `DeprecationWarning`.
```
**Bad** (new name first → `PipelineWorker`, a current class, gets flagged):
```
- ⚠️ `PipelineWorker` is the new name for `PipelineTask`.
```

## The deprecations.json registry

`deprecations.json` (repo root) is the machine-readable mirror of every
deprecation/removal — the form `check_deprecation` consumes directly, with no
prose parsing, so none of the phrasing failure modes can reach it. It is a
living file: add to it in the same PR that deprecates/removes an API. (It was
seeded by a one-time backfill of the release-note history; see its `_meta`
block.)

Append one object to `entries` **per deprecated/removed thing** (a PR that
removes `A`, `B`, and `C` adds three objects; a deprecated init param is one
object whose `subject` is the member, not the class):

```json
{
  "subject": "PipelineTask",
  "subject_kind": "symbol",
  "owner": null,
  "status": "deprecated",
  "deprecated_in": null,
  "removed_in": null,
  "reason": "rename",
  "replacement": { "relation": "rename", "targets": ["PipelineWorker"] },
  "migration": "The old name still resolves but emits a DeprecationWarning.",
  "source_version": null,
  "confidence": "high"
}
```

Field reference (full schema is in `deprecations.json` → `_meta.record_schema`):

- `subject` — the deprecated/removed thing; dotted `Owner.member` for members.
- `subject_kind` — `symbol`, `module`, `param`, `field`, `method`, `event`,
  `alias`, `extra`, or `nested_class`.
- `owner` — owning class for member kinds (stays current); `null` otherwise.
  **This is the field that stops a current class being reported deprecated.**
- `status` — `deprecated` or `removed` (match the `.md` type).
- `deprecated_in` / `removed_in` — set the one matching `status` to the release
  this ships in if known, else `null` (it can be stamped at release time).
- `reason` — `rename`, `move`, `split`, `merged`, `use_existing`,
  `behavior_default`, `unmaintained`, `vendor_gone`, `alias`, or `other`.
- `replacement.relation` — `none`, `rename`, `move`, `split`, `merged_into`, or
  `use_existing`. `replacement.targets` is `[]` when `relation` is `none`, and
  **never** contains the subject.
- `migration` — short how-to, or `null`.
- `source_version` — `null` for hand-added entries (set only on backfilled ones).
- `confidence` — `high` for hand-authored entries.

Every symbol in `subject` and `replacement.targets` must be a real Pipecat
symbol in this PR's diff — do not invent or normalize names.
