# Review a docs update

Instructions for a second pass over a finished `/update-docs` result, run in a
fresh session that did not write it. The writer has already committed to its
reading of the change; a reviewer that starts from the code catches what that
reading got wrong. The update-docs workflows run this after the writer step.

The caller provides:

- **The source PR**: its number and repository. The current directory is that
  repository at the PR's merge commit.
- **`DOCS_PATH`**: the docs repository, on the branch holding the update.
- **`BASE`**: the docs commit the update started from. The update is
  everything in `git -C DOCS_PATH diff BASE`.

You did not write the update. Your job is to find what it got wrong and fix it.

## Read first

- `DOCS_PATH/STYLE.md`, in full. It is the standard for how pages read and how
  large each kind of edit should be.
- The source PR: `gh pr view <number> --json title,body` and
  `gh pr diff <number>`. The PR description and its changelog fragments are
  the author's claims, and they can be wrong. The code at HEAD is what is true.
- The update: `git -C DOCS_PATH diff BASE`.

## Check, in this order

1. **Claims.** For every factual statement the update adds or changes, find the
   code that makes it true: read the source files at HEAD. A statement the code
   does not support is wrong even when the PR description or a changelog
   fragment says it. Fix it or remove it.
2. **Coverage.** List each reader-facing change in the code diff (something a
   user can set, call, or observe, or must do differently) and each bullet of
   any Breaking Changes section. Each one is either documented by the update,
   already correctly documented, or invisible to users. Add what is missing,
   briefly, in the page's existing format.
3. **Other pages.** Grep every `.mdx` file in `DOCS_PATH` for each class,
   parameter, event, environment variable, and behavior the change touches.
   For each hit outside the edited lines, decide whether the change made it
   false: names, defaults, lists of sibling classes, comparison tables,
   migration tables, and code samples. For a code sample, trace what it does
   under the new behavior: what runs, what fires, what the user sees or hears.
   Fix what is now wrong.
4. **Each edited page, read in full.** No fact stated twice. No narration of
   the change, maintainer rationale, or implementation detail a user doesn't
   need. No rewording of lines the change didn't make false: restore those to
   their `BASE` wording. Tables stay well formed (escape `|` inside a cell as
   `\|`). Examples use the current API.
5. **Size.** STYLE.md's "Changing Existing Pages" table gives the expected size
   of each kind of edit. Cut what exceeds it.

If the update should not exist at all, because nothing a reader needs has
changed, restore every file so that `git -C DOCS_PATH diff BASE` is empty and
say why.

## Rules

- Edit only files inside `DOCS_PATH`.
- Make the smallest fix that resolves each problem. Do not rewrite text that is
  correct, and do not add anything the checks above don't call for.
- Do not run Prettier and do not regenerate `llms.txt`.
- Commit your fixes on the update's branch. If you found nothing to fix, commit
  nothing.
- Finish with one line per problem found and the fix you made, or
  `No problems found.`
