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
