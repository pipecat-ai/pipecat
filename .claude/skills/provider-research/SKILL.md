---
name: provider-research
description: Research every provider behind Pipecat's services for new models and API affordances, writing per-service reports and local branches for clear-cut updates; publishing is scripts/provider-watch/publish.py's job, run outside this skill
disable-model-invocation: true
argument-hint: "[--only a,b] [--limit N] [--concurrency N]"
---

Run a provider-research sweep: one researcher subagent per service unit, a concise dated report per unit, a local digest, and a committed branch for every change a researcher is confident about. Everything stays local — this skill publishes nothing. Pushing reports, opening draft PRs on pipecat and filing the digest issue are `scripts/provider-watch/publish.py`'s job: the weekly workflow runs it in dedicated steps after each research job, and a local run ends by printing the commands for the invoker. You are the orchestrator; the research itself happens in `provider-watch-researcher` subagents following `RESEARCH_GUIDE.md`.

## Arguments

```
/provider-research [--only a,b] [--limit N] [--concurrency N]
```

- `--only a,b` — providers or unit ids (`openai`, `deepgram/stt`). Default: every unit.
- `--limit N` — research only the first N selected units (deterministic order). For test runs.
- `--concurrency N` — researchers per batch. Default 6; use 1 for a linear test run.

Examples:

- `/provider-research --only deepgram,groq --limit 2 --concurrency 1` — smoke test
- `/provider-research --only groq` — exercise the branch path; review the branch with the command the report prints
- `/provider-research --only <group>` — what one matrix job of the weekly workflow runs

## Instructions

### Step 1: Resolve paths and prerequisites

1. Parse the arguments. Record `RUN_DATE` as today's date (`YYYY-MM-DD`) and `PIPECAT_COMMIT` as `git rev-parse --short HEAD`.
2. Pick a scratch directory outside the repo (your session scratchpad if you have one, else `mktemp -d -t provider-research`). Everything transient — payloads, `run.jsonl`, worktrees — lives there.
3. Reports checkout: always `./_reports` in this repo (gitignored). If it is missing, `gh repo clone pipecat-ai/provider-watch-reports _reports`; if the clone fails, `git init _reports` and continue with no history. If it exists and has a remote, `git -C _reports pull --ff-only` so the run reads current memory.
4. Stop with a clear error if `uv run python scripts/provider-watch/inventory.py --md` fails.
5. Decision intake: the team records decisions as comments on the digest issues; researchers fold them into each unit's `decisions.md` in `_reports`. Collect the comments of the three most recent issues into `<scratch>/digest-comments.md`:
   ```bash
   gh issue list --repo pipecat-ai/provider-watch-reports --state all --search "Provider watch in:title sort:created-desc" --limit 3 --json number,title,url \
     | jq -r '.[].number' | while read -r n; do
       gh issue view "$n" --repo pipecat-ai/provider-watch-reports --json title,url,comments \
         --jq '"## \(.title) — \(.url)\n" + ([.comments[] | "- \(.author.login) (\(.createdAt | .[:10])) <\(.url)>:\n  \(.body | gsub("\n"; "\n  "))"] | join("\n"))'
     done > <scratch>/digest-comments.md
   ```
   If the repo or `gh` is unavailable, write an empty file. Every researcher gets the same file and picks out what concerns its unit.

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
  "decisions_file": "<reports_path>/reports/<provider>/<unit-suffix>/decisions.md",
  "digest_comments_file": "<scratch>/digest-comments.md",
  "scratch_dir": "<scratch>"
}
```

`<unit-suffix>` is the part of the unit id after the slash (`tts`, `responses-llm`). `report_path` is the repo-relative path used in frontmatter and links; `report_file` is where the researcher writes, spelled out absolutely so there is nothing to resolve. The previous report is the newest date-named file in that directory (`decisions.md` is not a report); pass `null` on a first run. `decisions_file` may not exist yet — the researcher creates it when it first records a decision.

Rules for the batch loop:

- Launch the whole batch at once so the subagents run concurrently; wait for all of them before starting the next batch.
- Researchers only produce local artifacts: the report, the unit's `decisions.md` when a comment or PR state decided something, and at most one committed `provider-watch/*` branch in a worktree under `<scratch>`. They never push or open PRs.
- Each researcher returns exactly one JSON line: `{"service", "default_model", "prs", "gaps", "error", "summary", "report_path"}`. Append it to `<scratch>/run.jsonl`. If a researcher fails or returns nothing usable, write the report yourself from `REPORT_TEMPLATE.md` with `error` set to what happened (no secrets), and append a matching line.
- If `git status` in this checkout shows changes you did not make, stop and report it.

### Step 4: Highlights

Write up to 5 highlight bullets to `<scratch>/highlights.md` from `run.jsonl` — for a small `--only` slice, only the 1–3 that must surface, since the workflow concatenates every group's bullets into one digest: what a maintainer should look at first (branches to review, long-open gaps, providers that errored). Skip bullets when nothing stands out.

### Step 5: Render the digest

```bash
uv run python scripts/provider-watch/digest.py --reports _reports --date <RUN_DATE> --highlights <scratch>/highlights.md --out _reports/digests/<RUN_DATE>.md
```

This is the local preview of what publishing would file as the digest issue.

### Step 6: Clean up and summarize

1. `git worktree prune` in this checkout and remove `<scratch>/wt-*` directories. Branches stay; they are the run's output.
2. Print a summary table — unit, default model, branch, changes to consider, error — plus the digest path (`_reports/digests/<RUN_DATE>.md`) and the review command for each branch (`git show <branch>`).
3. End with how to publish, since this skill never does: `uv run python scripts/provider-watch/publish.py --date <RUN_DATE>` pushes the branches, opens the draft PRs and pushes the reports; adding `--finalize --highlights <scratch>/highlights.md` also files the digest issue. Publishing is the invoker's call.

## Unattended runs

The weekly workflow fans the sweep out: a plan job slices the units into groups (`scripts/provider-watch/plan.py`), one matrix job per group runs `--only <group>`, and dedicated workflow steps — not this skill — run `publish.py` over what each group left behind, with a final digest job filing the issue. Never ask anything (nothing in this skill needs asking); a researcher failure never aborts the run.

## Guardrails

- Never print, commit, or paste environment variable values, `Authorization` headers, or raw API keys — in reports or your output. `probe.py` redacts; ad-hoc output must be checked by hand.
- This skill publishes nothing: never push, never open PRs or issues, never run `publish.py` — print its commands instead. Researchers follow the same rule.
- Only `scripts/provider-watch/*`, `RESEARCH_GUIDE.md` and `REPORT_TEMPLATE.md` define what a researcher does; do not improvise extra instructions per unit beyond the payload.
