---
name: provider-watch
description: Research every provider behind Pipecat's services for new models and API affordances, write per-service reports to the provider-watch reports repo, and propose draft PRs for clear-cut updates
disable-model-invocation: true
argument-hint: "[--only a,b] [--limit N] [--concurrency N] [--publish] [--non-interactive]"
---

Run a provider-research sweep: one researcher subagent per service unit, a concise dated report per unit, a digest, and a branch for every change a researcher is confident about. All of that is produced locally first; publishing — pushing reports, opening draft PRs on pipecat, filing the digest issue — happens through `scripts/provider-watch/publish.py`, either as the run goes (`--publish`) or after the maintainer confirms at the end. You are the orchestrator; the research itself happens in `provider-watch-researcher` subagents following `RESEARCH_GUIDE.md`.

## Arguments

```
/provider-watch [--only a,b] [--limit N] [--concurrency N] [--publish] [--non-interactive]
```

- `--only a,b` — providers or unit ids (`openai`, `deepgram/stt`). Default: every unit.
- `--limit N` — research only the first N selected units (deterministic order). For test runs.
- `--concurrency N` — researchers per batch. Default 6; use 1 for a linear test run.
- `--publish` — publish as the run goes, without asking. Without it, the run is a dry run until Step 5 asks whether to publish; the default answer is no.
- `--non-interactive` — never ask anything; an unasked question takes its default. Without `--publish` this is an unattended dry run. Also fails fast if prerequisites are missing.

Examples:

- `/provider-watch --only deepgram,groq --limit 2 --concurrency 1` — smoke test; asks at the end, default no
- `/provider-watch --only groq` — exercise the branch path; review the branch with the command the report prints
- `/provider-watch --publish --non-interactive` — what the weekly workflow runs

## Instructions

### Step 1: Resolve paths and prerequisites

1. Parse the arguments. Record `RUN_DATE` as today's date (`YYYY-MM-DD`) and `PIPECAT_COMMIT` as `git rev-parse --short HEAD`.
2. Pick a scratch directory outside the repo (your session scratchpad if you have one, else `mktemp -d -t provider-watch`). Everything transient — payloads, `run.jsonl`, worktrees — lives there.
3. Reports checkout: always `./_reports` in this repo (gitignored). If it is missing, `gh repo clone pipecat-ai/provider-watch _reports`; if the clone fails without `--publish` (repo not created yet), `git init _reports` and continue with no history. With `--publish`, run `git -C _reports pull --ff-only` first.
4. With `--publish` or `--non-interactive`, stop with a clear error if any of these fails: `gh auth status`, `_reports` exists and is on `main` (publish only), `uv run python scripts/provider-watch/inventory.py --md`.

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
  "reports_path": "<absolute path of ./_reports>",
  "report_path": "reports/<provider>/<unit-suffix>/<RUN_DATE>.md",
  "report_file": "<reports_path>/reports/<provider>/<unit-suffix>/<RUN_DATE>.md",
  "previous_report_file": "<absolute path of the newest existing reports/<provider>/<unit-suffix>/*.md, or null>",
  "scratch_dir": "<scratch>"
}
```

`<unit-suffix>` is the part of the unit id after the slash (`tts`, `responses-llm`). `report_path` is the repo-relative path used in frontmatter and links; `report_file` is where the researcher writes, spelled out absolutely so there is nothing to resolve. The previous report is the newest dated file in that directory; pass `null` on a first run.

Rules for the batch loop:

- Launch the whole batch at once so the subagents run concurrently; wait for all of them before starting the next batch.
- Researchers only produce local artifacts: the report, and at most one committed `provider-watch/*` branch in a worktree under `<scratch>`. They never push or open PRs.
- Each researcher returns exactly one JSON line: `{"service", "status", "default_model", "prs", "summary", "report_path"}`. Append it to `<scratch>/run.jsonl`. If a researcher fails or returns nothing usable, write the report yourself from `REPORT_TEMPLATE.md` with `status: error` and the failure in the body, and append a matching line.
- **With `--publish`:** after every batch run `uv run python scripts/provider-watch/publish.py --date <RUN_DATE>`. It pushes the finished units' branches, opens their draft PRs (up to 8 per run; the rest stay branches marked `capped`), rewrites their reports with the PR URLs, and pushes `_reports`. It is idempotent, so a run that dies keeps everything published so far and a re-run picks up the rest.
- If `git status` in this checkout shows changes you did not make, stop and report it.

### Step 4: Highlights

Write 3–5 highlight bullets to `<scratch>/highlights.md` from `run.jsonl`: what a maintainer should look at first (PRs or branches to review, defaults that look stale, providers that errored). Skip bullets when nothing stands out.

### Step 5: Publish or ask

- **With `--publish`:** `uv run python scripts/provider-watch/publish.py --date <RUN_DATE> --finalize --highlights <scratch>/highlights.md`. This renders `digests/<RUN_DATE>.md`, pushes it, and opens the digest issue on the reports repo (or updates it on a re-run) when anything is worth showing.
- **Without `--publish`, interactive:** render the digest locally first — `uv run python scripts/provider-watch/digest.py --reports _reports --date <RUN_DATE> --highlights <scratch>/highlights.md --out _reports/digests/<RUN_DATE>.md` — then ask exactly one question, with **"No — keep everything local"** as the first (default) option and the publish option spelling out the scope: "Publish: push N reports and the digest to pipecat-ai/provider-watch, push M branches and open M draft PRs on pipecat-ai/pipecat, open the digest issue." Only an explicit choice of the publish option publishes; any other answer, no answer, or an interrupted session means no. If yes, run the `--finalize` command above.
- **Without `--publish`, `--non-interactive`:** render the digest locally as above and publish nothing.

### Step 6: Clean up and summarize

1. `git worktree prune` in this checkout and remove `<scratch>/wt-*` directories. Branches stay; they are the dry-run output.
2. Print a summary table — unit, status, default model, PR or branch — plus the digest path (`_reports/digests/<RUN_DATE>.md`), the review command for each branch (`git show <branch>`), and, when nothing was published, how to publish later: re-run with `--publish` (the local reports are the baseline, so it is cheap) or run `publish.py --date <RUN_DATE> --finalize` by hand.

## Unattended runs

The weekly workflow runs `--publish --non-interactive`; its dry-run input runs `--non-interactive` alone. In those modes never ask anything — every decision above has a default — and exit non-zero only when Step 1 prerequisites fail; a researcher failure never aborts the run. The workflow sets the bot git identity before the skill runs.

## Guardrails

- Never print, commit, or paste environment variable values, `Authorization` headers, or raw API keys — in reports, PR bodies, issues, or your output. `probe.py` redacts; ad-hoc output must be checked by hand.
- Only `publish.py` pushes or opens anything, and only in Step 3 (with `--publish`) and Step 5. Never push to `pipecat-ai/pipecat` `main`, never force-push, never close or merge PRs.
- Only `scripts/provider-watch/*`, `RESEARCH_GUIDE.md` and `REPORT_TEMPLATE.md` define what a researcher does; do not improvise extra instructions per unit beyond the payload.
