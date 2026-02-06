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

4. Each changelog file should at least contain a main single line starting with `- ` followed by a clear description of the change.

5. If the change is complicated, changelog files can have indented lines after the main line with additional details or code samples.

6. Use ⚠️ emoji prefix for breaking changes.

## Example

For PR #3519 with a new feature and a bug fix:

`changelog/3519.added.md`:
```
- Added `SomeNewFeature` for doing something useful.
```

`changelog/3519.fixed.md`:
```
- Fixed an issue where something was not working correctly.
```
