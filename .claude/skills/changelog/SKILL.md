---
name: changelog
description: Create changelog files for important commits in a PR
---

Create changelog files for the important commits in this PR. The PR number is provided as an argument.

## Instructions

1. First, check what commits are on the current branch compared to main:
   ```
   git log main..HEAD --oneline
   ```

2. For each significant change, create a changelog file in the `changelog/` folder using the format:
   - `{PR_NUMBER}.added.md` - for new features
   - `{PR_NUMBER}.added.2.md`, `{PR_NUMBER}.added.3.md` - for additional new features
   - `{PR_NUMBER}.changed.md` - for changes to existing functionality
   - `{PR_NUMBER}.fixed.md` - for bug fixes
   - `{PR_NUMBER}.deprecated.md` - for deprecations

3. Each changelog file should at least contain a main single line starting with `- ` followed by a clear description of the change.

4. If the change is complicated, changelog files can have indented lines after the main line with additional details or code samples.

5. Use ⚠️ emoji prefix for breaking changes.

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
