---
name: pr-description
description: Update a GitHub PR description with a summary of changes
---

Update a GitHub pull request description based on the changes in the PR.

## Arguments

```
/pr-description <PR_NUMBER> [--fixes <ISSUE_NUMBERS>]
```

- `PR_NUMBER` (required): The pull request number to update
- `--fixes` (optional): Comma-separated issue numbers that this PR fixes (e.g., `--fixes 123,456`)

Examples:
- `/pr-description 3534`
- `/pr-description 3534 --fixes 123`
- `/pr-description 3534 --fixes 123,456,789`

## Instructions

1. First, gather information about the PR:
   - Use GitHub plugin to get PR details (title, current description, base branch)
   - Use local git to get commits: `git log main..HEAD --oneline`
   - Use local git to get the diff: `git diff main..HEAD`
   - Parse any `--fixes` argument for issue numbers

2. Check the existing PR description:
   - If it already has a complete, accurate description that reflects the changes, do nothing
   - If it's missing sections, incomplete, or outdated compared to the actual changes, proceed to update
   - If it only has the template placeholder text, generate a full description

3. Analyze the changes:
   - Understand the purpose of each commit
   - Identify any breaking changes (API changes, removed features, behavior changes)
   - Look for new features, bug fixes, refactoring, or documentation changes
   - Collect issue numbers from:
     - The `--fixes` argument (if provided)
     - Commit messages (patterns like "Fixes #123", "Closes #456", "Resolves #789")

4. Generate or update the PR description with these sections:

## PR Description Format

### Summary (always include)

Brief bullet points describing what changed and why. Focus on the *purpose* and *impact*, not implementation details.

```markdown
## Summary

- Added X to enable Y
- Fixed bug where Z would happen
- Refactored W for better maintainability
```

### Breaking Changes (include only if applicable)

Document any changes that affect existing users or APIs.

```markdown
## Breaking Changes

- `ClassName.method()` now requires a `param` argument
- Removed deprecated `old_function()` - use `new_function()` instead
```

### Testing (include when non-obvious)

How to verify the changes work. Skip for trivial changes.

```markdown
## Testing

- Run `uv run pytest tests/test_feature.py` to verify the fix
- Example usage: `uv run examples/new_feature.py`
```

### Fixes (include if issues are provided or found in commits)

List issues this PR fixes. GitHub will automatically close these issues when the PR is merged.

```markdown
## Fixes

- Fixes #123
- Fixes #456
```

Note: Use "Fixes #X" format (not "Closes" or "Resolves") for consistency. Each issue should be on its own line with "Fixes" to ensure GitHub auto-closes them.

## Guidelines

- **Be concise** - Reviewers should understand the PR in 30 seconds
- **Focus on why** - The diff shows *what* changed, explain *why*
- **Skip empty sections** - Only include sections that have content
- **Use bullet points** - Easier to scan than paragraphs
- **Don't duplicate the diff** - Avoid listing every file or line changed

## Example Output

```markdown
## Summary

- Added `/docstring` skill for documenting Python modules with Google-style docstrings
- Skill finds classes by name and handles conflicts when multiple matches exist
- Skips already-documented code to avoid unnecessary changes

## Testing

/docstring ClassName

## Fixes

- Fixes #123
```

## Checklist

Before updating the PR:

- [ ] Verified existing description needs updating (not already complete)
- [ ] Summary accurately reflects the changes
- [ ] Breaking changes are clearly documented (if any)
- [ ] No unnecessary sections included
- [ ] Description is concise and scannable
