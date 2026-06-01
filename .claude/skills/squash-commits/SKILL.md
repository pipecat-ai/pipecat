---
name: squash-commits
description: Reorganize messy branch commits into a small set of logical, meaningful commits without changing any content. Drops merge-from-main commits. Safe: creates a backup branch first.
---

Reorganize the commits on the current branch into a small number of logical commits. Do NOT change any file content — only the commit structure changes.

## Instructions

### 1. Safety check

```bash
git status --short
```

If there are uncommitted changes, stop and tell the user to commit or stash them first.

### 2. Inspect the branch

```bash
git log main..HEAD --oneline
git diff main..HEAD --name-only
```

List every file changed vs `main` and every commit on the branch (excluding merge commits from main).

### 3. Create a backup branch

```bash
git branch backup/<current-branch-name>
```

Tell the user the backup exists so they can recover if needed.

### 4. Soft-reset to main and unstage everything

```bash
git reset --soft main
git restore --staged .
```

All branch changes are now in the working tree, unstaged. No content has changed.

### 5. Plan the logical groups

Read the changed files and the original commit messages to understand what the work covers. Group related files into logical commits. Typical groups:

- Core feature or fix (new source files + modified core files)
- Secondary features or fixes (each as its own commit if distinct)
- Refactoring or renames
- Tests
- Changelogs / docs

Use the changelog files (if any) as a strong hint — each changelog entry often maps to one commit.

Present the proposed grouping to the user and ask for confirmation before committing.

### 6. Commit in logical groups

For each group, stage only the relevant files and commit with a clear message following the project's conventions:

```bash
git add <file1> <file2> ...
git commit -m "..."
```

Use conventional commit prefixes if the project uses them (`feat:`, `fix:`, `refactor:`, `test:`, `chore:`).

### 7. Verify

```bash
git log main..HEAD --oneline
git diff main..HEAD --name-only
git status --short
```

Confirm:
- Commit count is small and each message is meaningful
- The set of changed files vs `main` is identical to before
- Working tree is clean

### 8. Remind about force-push

The branch history has been rewritten. Tell the user they will need to `git push --force-with-lease` when they are ready to update the remote. Do NOT push automatically.

## Rules

- Never change file contents. If you find yourself editing a file, stop.
- Never skip the backup branch step.
- Never force-push without explicit user instruction.
- If any step fails or the result looks wrong, tell the user and suggest restoring from the backup: `git reset --hard backup/<branch-name>`.
