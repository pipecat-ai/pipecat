# Changelog Fragments

This directory contains changelog fragments that will be compiled into the main `CHANGELOG.md` at release time using [towncrier](https://towncrier.readthedocs.io/).

## For Contributors: Adding a Changelog Entry

When you make a change that should be documented in the changelog, create a new file in this directory.

### Creating a Fragment

Create a file in this directory with this naming pattern:

```
changelog/<PR_number>.<type>.md
```

Choose the appropriate type:

- `added.md` - New features
- `changed.md` - Changes in existing functionality
- `deprecated.md` - Soon-to-be removed features
- `removed.md` - Removed features
- `fixed.md` - Bug fixes
- `security.md` - Security fixes

Write your changelog entry as a Markdown bullet point. Include the `-` at the start:

**Example files:**

`changelog/1234.added.md`:

```markdown
- Added support for Anthropic Claude 3.5 Sonnet with improved streaming performance.
```

`changelog/5678.fixed.md`:

```markdown
- Fixed an issue where audio frames were dropped during high-load scenarios.
```

**For entries with nested bullets:**

`changelog/1234.changed.md`:

```markdown
- Updated service configuration:

  - Changed default timeout to 30 seconds
  - Added retry logic for failed connections
  - Improved error messages
```

### Multiple Changes in One PR

**Different types of changes:** Create separate fragment files for each type:

```
changelog/1234.added.md
changelog/1234.fixed.md
```

**Multiple changes of the same type:** Create numbered fragment files:

```
changelog/1234.changed.md
changelog/1234.changed.2.md
changelog/1234.changed.3.md
```

**Related changes:** Use nested bullets in a single fragment:

```markdown
- Updated service configuration:

  - Changed default timeout to 30 seconds
  - Added retry logic for failed connections
```

**Rule of thumb:** One logical change per fragment file. If changes are unrelated, use separate files.

### Preview Your Changes

To see what your changelog entry will look like:

```bash
towncrier build --draft --version Unreleased
```

This won't modify any files, just show you a preview.

### When to Skip Changelog Entries

You can skip adding a changelog entry for:

- Documentation-only changes
- Internal refactoring with no user-facing impact
- Test-only changes
- CI/build configuration changes

If you're unsure whether your change needs a changelog entry, ask in your PR!

---

## For Maintainers: Releasing

### Automated Release

The changelog is automatically generated via GitHub Actions. When you're ready to release:

1. **Trigger the workflow** from GitHub Actions:

   - Go to Actions → "Generate Changelog for Release"
   - Click "Run workflow"
   - Enter the version (e.g., `0.0.97`)
   - Enter the release date (e.g., `2025-12-04`)

2. **Review the PR** that gets created automatically

3. **Merge the PR** to update the changelog

## References

- [Towncrier Documentation](https://towncrier.readthedocs.io/)
- [Keep a Changelog](https://keepachangelog.com/)
- [Contributing Guide](../CONTRIBUTING.md)
