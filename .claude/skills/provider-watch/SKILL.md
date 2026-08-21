---
name: provider-watch
description: Research every provider behind Pipecat's services for new models and API affordances, write per-service reports to the provider-watch reports repo, and open draft PRs for clear-cut updates
disable-model-invocation: true
argument-hint: "[--only a,b] [--limit N] [--concurrency N] [--no-prs] [--no-probe] [--publish|--local] [--reports-path P] [--ci]"
---

Run a provider-research sweep: one researcher subagent per service unit, a concise dated report per unit in the reports repo, a digest, and draft PRs on pipecat for changes the researcher is confident about. You are the orchestrator; the research itself happens in `provider-watch-researcher` subagents following `RESEARCH_GUIDE.md`.

## Arguments

```
/provider-watch [--only a,b] [--limit N] [--concurrency N] [--no-prs] [--no-probe] [--publish|--local] [--reports-path P] [--ci]
```

- `--only a,b` — providers or unit ids (`openai`, `deepgram/stt`). Default: every unit.
- `--limit N` — research only the first N selected units (deterministic order). For test runs.
- `--concurrency N` — researchers per batch. Default 6; use 1 for a linear test run.
- `--no-prs` — never create branches or PRs. Changes that meet the PR criteria are written up under "PRs withheld" (status `prs-withheld`) so a test run shows what a real run would have opened.
- `--no-probe` — skip live provider calls (`probe.py`, ad-hoc scripts, evals). Implies `--no-prs`: a PR needs a passing probe.
- `--publish` / `--local` — push reports, open PRs and the digest issue / write everything locally and push nothing. Without either, ask once (Step 1).
- `--reports-path P` — existing checkout of the reports repo. Default: `./_reports` if present, else a fresh clone in the scratch dir.
- `--ci` — unattended mode: never ask questions, implies `--publish`, fail fast if prerequisites are missing.

Examples:

- `/provider-watch --only deepgram,groq --limit 2 --concurrency 1 --no-prs` — cheap local smoke test
- `/provider-watch --only ollama --local` — exercise the PR path without pushing
- `/provider-watch --ci --reports-path ./_reports` — what the weekly workflow runs

## Instructions

### Step 1: Resolve mode and paths

1. Parse the arguments. Record `RUN_DATE` as today's date (`YYYY-MM-DD`) and `PIPECAT_COMMIT` as `git rev-parse --short HEAD`.
2. Pick a scratch directory outside the repo (your session scratchpad if you have one, else `mktemp -d -t provider-watch`). Everything transient — payloads, `run.jsonl`, worktrees — lives there.
3. Mode: `--ci` or `--publish` ⇒ **publish**; `--local` ⇒ **local-only**. Otherwise ask the user exactly one question: *"Write reports locally only (default — no push, no PRs, no issue; PR branches are created but not pushed), or publish?"* Treat no answer as local-only.
4. Reports checkout: `--reports-path`, else `./_reports` if it exists, else `gh repo clone pipecat-ai/provider-watch <scratch>/provider-watch-reports`. If the clone fails in local-only mode (repo not created yet), `git init <scratch>/provider-watch-reports` and continue with no history. In publish mode run `git pull --ff-only` first.
5. In `--ci` mode, stop with a clear error if any of these is missing: `gh auth status` succeeds, the reports checkout exists and is on `main`, `uv run python scripts/provider-watch/inventory.py --md` runs.

### Step 2: Build the unit list

```bash
uv run python scripts/provider-watch/inventory.py --json [--only ...] [--limit N] > <scratch>/units.json
```

Each entry is one research unit (`id` like `cartesia/tts`) with its classes, default model, settings fields, thin-wrapper flag, registry/env/example-bot pointers and docs URL. Do not hand-edit or re-derive this; the researcher gets the entry verbatim.

### Step 3: Research in batches

Process units in `--concurrency`-sized batches, in the order `inventory.py` emits them. For each unit in a batch, launch one **`provider-watch-researcher`** subagent with this payload in the prompt. The agent is defined for Claude Code in `.claude/agents/provider-watch-researcher.md` (Agent tool, `subagent_type: provider-watch-researcher`) and for Codex in `.codex/agents/provider-watch-researcher.toml` (spawn the `provider-watch-researcher` agent); in an agent without subagents, do the researcher's work yourself, one unit at a time, by following `RESEARCH_GUIDE.md` with the same payload — the agent definitions are thin shims over that guide.

```json
{
  "unit": <the inventory entry>,
  "run_date": "<RUN_DATE>",
  "pipecat_commit": "<PIPECAT_COMMIT>",
  "repo_root": "<absolute path of this checkout>",
  "reports_path": "<absolute path of the reports checkout>",
  "report_path": "reports/<provider>/<unit-suffix>/<RUN_DATE>.md",
  "report_file": "<reports_path>/reports/<provider>/<unit-suffix>/<RUN_DATE>.md",
  "previous_report_file": "<absolute path of the newest existing reports/<provider>/<unit-suffix>/*.md, or null>",
  "scratch_dir": "<scratch>",
  "mode": "publish" | "local-only",
  "prs_enabled": true | false,
  "probe_enabled": true | false,
  "pr_budget_remaining": <int>
}
```

`<unit-suffix>` is the part of the unit id after the slash (`tts`, `responses-llm`). `report_path` is the repo-relative path used in frontmatter and links; `report_file` is where the researcher writes, spelled out absolutely so there is nothing to resolve. The previous report is the newest dated file in that directory; pass `null` on a first run.

Rules for the batch loop:

- Launch the whole batch at once so the subagents run concurrently; wait for all of them before starting the next batch.
- Each researcher returns exactly one JSON line: `{"service", "status", "default_model", "prs", "summary", "report_path"}`. Append it to `<scratch>/run.jsonl`. If a researcher fails or returns nothing usable, write the report yourself from `REPORT_TEMPLATE.md` with `status: error` and the failure in the body, and append a matching line.
- PR budget: at most 1 PR per unit and 8 per run. Pass `pr_budget_remaining` = 8 − PRs opened so far; once it reaches 0 pass `prs_enabled: false` to later batches (their qualifying changes land under "PRs withheld").
- **Publish mode:** after every batch, commit and push the reports checkout (`git add reports && git commit -m "provider-watch: <RUN_DATE> (<unit ids>)" && git push`). A run that dies later keeps what it has done.
- Researchers never touch this checkout's git state; PR work happens in worktrees under `<scratch>`. If `git status` here shows changes you did not make, stop and report it.

### Step 4: Digest

1. Write 3–5 highlight bullets to `<scratch>/highlights.md` from `run.jsonl`: what a maintainer should look at first (PRs to review, defaults that look stale, providers that errored). Skip bullets when nothing stands out.
2. Render:
   ```bash
   uv run python scripts/provider-watch/digest.py --reports <reports_path> --date <RUN_DATE> \
     --highlights <scratch>/highlights.md --out <reports_path>/digests/<RUN_DATE>.md
   ```
3. Publish mode: commit and push the digest.

### Step 5: Notify (publish mode only)

If any report has a status other than `up-to-date`, or any PR was opened, open one issue on the reports repo:

```bash
gh issue create --repo pipecat-ai/provider-watch --title "Provider watch <RUN_DATE>" \
  --body-file <reports_path>/digests/<RUN_DATE>.md
```

If an issue with that title already exists (re-run), update its body with `gh issue edit` instead. If everything is up to date, do not open an issue; print `No provider changes found.`

### Step 6: Clean up and summarize

1. `git worktree prune` in this checkout, and remove `<scratch>/wt-*` directories (local-only mode keeps the branches; the worktrees can go).
2. Print a summary table — unit, status, default model, PRs — and the paths of the digest and any local branches. In local-only mode remind the user nothing was pushed.

## Unattended (`--ci`) behaviour

The weekly workflow invokes this skill with `--ci`. In that mode:

- Never ask questions; every decision has a default above.
- Set the git identity in the reports checkout and worktrees to `github-actions[bot]` / `github-actions[bot]@users.noreply.github.com` before committing.
- Re-runs are safe: reports for today are overwritten, PR branches that already exist on origin are left alone and referenced, the digest issue is edited rather than duplicated.
- A researcher failure never aborts the run. Exit non-zero only when Step 1 prerequisites fail.

## Guardrails

- Never print, commit, or paste environment variable values, `Authorization` headers, or raw API keys — in reports, PR bodies, issues, or your output. `probe.py` redacts; ad-hoc output must be checked by hand.
- Never push to `pipecat-ai/pipecat` `main`, never force-push, never close or merge PRs.
- Only `scripts/provider-watch/*`, `RESEARCH_GUIDE.md` and `REPORT_TEMPLATE.md` define what a researcher does; do not improvise extra instructions per unit beyond the payload.
