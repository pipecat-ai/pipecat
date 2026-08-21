#!/usr/bin/env python3
#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Publish what a provider-watch run has produced locally.

Works entirely from disk — the reports for a date under ``_reports`` and the
local ``provider-watch/*`` branches they name — so it can run after every batch
of a publishing run, once at the end of a dry run, or by hand to finish a run
that died. Every step is idempotent: branches already on origin are not pushed
again, a branch with an open PR adopts that PR, reports already pointing at a
PR URL are left alone, and the digest issue is edited rather than duplicated.

For each report whose ``prs`` list has an entry in ``state: branch``:

1. push the branch and open a draft PR (title and body from the branch's
   commit message, plus a link to the report), subject to the per-run cap;
2. rewrite the report — frontmatter entry to ``state: open`` with the URL,
   and the body's branch/review line to the URL.

Corrections researchers recorded under ``hints`` (a replacement for a dead URL,
a spec worth tracking) are merged into ``providers.yaml`` on one branch per run
and opened as one draft PR outside the cap, so the file stays current without
anyone editing it by hand.

Then commit and push ``_reports``. With ``--finalize`` it also renders the
digest and opens (or updates) the digest issue on the reports repo when there
is anything to show. Run::

    uv run python scripts/provider-watch/publish.py --date 2026-08-20
    uv run python scripts/provider-watch/publish.py --date 2026-08-20 --finalize --highlights h.md
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import digest  # noqa: E402

REPO_ROOT = HERE.parents[1]
DEFAULT_REPORTS = REPO_ROOT / "_reports"
PROVIDERS_YAML = Path(".claude/skills/provider-watch/providers.yaml")
PR_LABEL = "provider-watch"
DEFAULT_PR_CAP = 8
HINT_SCALARS = ("models", "changelog", "notes")
HINT_LISTS = ("docs", "specs")

# The line a researcher writes under "## PRs" for a local branch; rewritten to
# the PR URL once the PR exists.
BRANCH_LINE = re.compile(
    r"^- `(?P<branch>provider-watch/[^`\s]+)` — review: `git show (?P=branch)`",
    re.MULTILINE,
)


class Shell:
    """Runs git and gh; tests swap in a fake."""

    def run(self, *args: str, cwd: Path | None = None, check: bool = True) -> str:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
        if check and result.returncode != 0:
            raise RuntimeError(
                f"{' '.join(args)}: {result.stderr.strip() or result.stdout.strip()}"
            )
        return result.stdout

    def ok(self, *args: str, cwd: Path | None = None) -> bool:
        return subprocess.run(args, cwd=cwd, capture_output=True).returncode == 0


@dataclass
class Report:
    path: Path
    meta: dict
    body: str

    @classmethod
    def load(cls, path: Path) -> Report:
        text = path.read_text()
        meta = digest.parse_frontmatter(text)
        end = text.find("\n---", 3)
        body = text[end + 4 :] if text.startswith("---") and end != -1 else text
        return cls(path, meta, body)

    def save(self) -> None:
        front = yaml.safe_dump(self.meta, sort_keys=False, allow_unicode=True, width=1000).rstrip()
        body = self.body if self.body.startswith("\n") else "\n" + self.body
        self.path.write_text(f"---\n{front}\n---{body}")


@dataclass
class Outcome:
    opened: list[str] = field(default_factory=list)
    adopted: list[str] = field(default_factory=list)
    capped: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    hints_pr: str | None = None
    reports_pushed: bool = False
    issue_url: str | None = None


def load_reports(reports_dir: Path, date: str) -> list[Report]:
    return [Report.load(p) for p in sorted(reports_dir.glob(f"reports/*/*/{date}.md"))]


def _open_pr_for_branch(sh: Shell, repo: str, branch: str) -> str | None:
    out = sh.run(
        "gh", "pr", "list", "--repo", repo, "--head", branch, "--state", "open", "--json", "url"
    )
    prs = json.loads(out or "[]")
    return prs[0]["url"] if prs else None


def _commit_message(sh: Shell, repo_root: Path, branch: str) -> tuple[str, str]:
    subject = sh.run("git", "log", "-1", "--format=%s", branch, cwd=repo_root).strip()
    body = sh.run("git", "log", "-1", "--format=%b", branch, cwd=repo_root).strip()
    return subject, body


def publish_prs(
    reports: list[Report],
    *,
    sh: Shell,
    repo_root: Path,
    pipecat_repo: str,
    reports_repo: str,
    date: str,
    cap: int,
) -> Outcome:
    """Open PRs for branch-state entries (up to ``cap`` per run) and rewrite the reports."""
    outcome = Outcome()
    opened_this_run = sum(
        1
        for r in reports
        for pr in r.meta.get("prs") or []
        if pr.get("state") == "open" and pr.get("opened") == date
    )

    for report in reports:
        changed = False
        for pr in report.meta.get("prs") or []:
            if pr.get("state") != "branch" or not pr.get("branch"):
                continue
            branch = pr["branch"]
            if not sh.ok("git", "rev-parse", "--verify", "--quiet", branch, cwd=repo_root):
                outcome.skipped.append(f"{branch}: no such local branch")
                continue

            url = _open_pr_for_branch(sh, pipecat_repo, branch)
            if url:
                outcome.adopted.append(url)
            elif opened_this_run >= cap:
                pr["capped"] = True
                outcome.capped.append(branch)
                changed = True
                continue
            else:
                sh.run("git", "push", "-u", "origin", branch, cwd=repo_root)
                subject, body = _commit_message(sh, repo_root, branch)
                report_path = report.path.relative_to(report.path.parents[3]).as_posix()
                pr_body = f"{body}\n\n## Report\n\nhttps://github.com/{reports_repo}/blob/main/{report_path}".strip()
                url = (
                    sh.run(
                        "gh",
                        "pr",
                        "create",
                        "--repo",
                        pipecat_repo,
                        "--draft",
                        "--label",
                        PR_LABEL,
                        "--head",
                        branch,
                        "--title",
                        subject,
                        "--body",
                        pr_body,
                        cwd=repo_root,
                    )
                    .strip()
                    .splitlines()[-1]
                )
                opened_this_run += 1
                outcome.opened.append(url)

            pr.update({"state": "open", "url": url, "opened": date})
            pr.pop("capped", None)
            report.body = BRANCH_LINE.sub(
                lambda m, b=branch, u=url: f"- {u}" if m.group("branch") == b else m.group(0),
                report.body,
            )
            changed = True

        if changed:
            report.save()
    return outcome


# --------------------------------------------------------------------- hints


def merge_hints(data: dict, provider: str, hints: dict) -> bool:
    """Fold one report's ``hints`` into the providers.yaml mapping; True if anything changed."""
    entry = data.get(provider) or {}
    changed = False
    for key in HINT_SCALARS:
        if hints.get(key) and hints[key] != entry.get(key):
            entry[key] = hints[key]
            changed = True
    for key in HINT_LISTS:
        for item in hints.get(key) or []:
            existing = entry.setdefault(key, [])
            if item not in existing:
                existing.append(item)
                changed = True
    if changed:
        data[provider] = entry
    return changed


class _IndentedDumper(yaml.SafeDumper):
    """Indents list items under their key, the way the file is written by hand."""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def _strip_strings(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return [_strip_strings(v) for v in value]
    if isinstance(value, dict):
        return {k: _strip_strings(v) for k, v in value.items()}
    return value


def render_providers_yaml(original: str, data: dict) -> str:
    """Header comment block kept verbatim; the mapping regenerated below it."""
    header_lines = []
    for line in original.splitlines():
        if line.startswith("#") or not line.strip():
            header_lines.append(line)
        else:
            break
    header = "\n".join(header_lines).rstrip() + "\n\n"
    body = yaml.dump(
        _strip_strings(data),
        Dumper=_IndentedDumper,
        sort_keys=False,
        allow_unicode=True,
        width=1000,
    )
    # One blank line between providers keeps the file readable.
    spaced = "\n".join(
        ("\n" + line) if (line and not line.startswith(" ") and i) else line
        for i, line in enumerate(body.rstrip().splitlines())
    )
    return header + spaced + "\n"


def publish_hints(
    reports: list[Report],
    *,
    sh: Shell,
    repo_root: Path,
    pipecat_repo: str,
    date: str,
    scratch: Path,
) -> str | None:
    """Turn the run's hint corrections into one draft PR against providers.yaml.

    Returns the PR URL, or None when no report proposed anything new.
    """
    proposals = [
        (r.meta["service"].split("/")[0], r.meta["hints"])
        for r in reports
        if isinstance(r.meta.get("hints"), dict) and r.meta.get("service")
    ]
    if not proposals:
        return None
    yaml_path = repo_root / PROVIDERS_YAML
    original = yaml_path.read_text()
    data = yaml.safe_load(original) or {}
    providers = sorted({p for p, h in proposals if merge_hints(data, p, h)})
    if not providers:
        return None

    branch = f"provider-watch/providers-{date}"
    existing = _open_pr_for_branch(sh, pipecat_repo, branch)
    if existing:
        return existing
    base = (
        "origin/main"
        if sh.ok("git", "rev-parse", "--verify", "--quiet", "origin/main", cwd=repo_root)
        else "main"
    )
    worktree = scratch / f"wt-providers-{date}"
    title = f"Update provider-watch hints for {', '.join(providers)}"
    sh.run("git", "worktree", "add", str(worktree), "-b", branch, base, cwd=repo_root)
    try:
        (worktree / PROVIDERS_YAML).write_text(render_providers_yaml(original, data))
        sh.run("git", "add", str(PROVIDERS_YAML), cwd=worktree)
        sh.run(
            "git",
            "commit",
            "-q",
            "-m",
            f"{title}\n\nReplacement URLs and specs the researchers recorded this run.",
            cwd=worktree,
        )
        sh.run("git", "push", "-u", "origin", branch, cwd=worktree)
        url = (
            sh.run(
                "gh",
                "pr",
                "create",
                "--repo",
                pipecat_repo,
                "--draft",
                "--label",
                PR_LABEL,
                "--head",
                branch,
                "--title",
                title,
                "--body",
                f"Replacement URLs and specs the provider-watch researchers recorded on {date}; "
                "each provider's report for that date says why.",
                cwd=worktree,
            )
            .strip()
            .splitlines()[-1]
        )
    finally:
        sh.run("git", "worktree", "remove", "--force", str(worktree), cwd=repo_root, check=False)
    return url


def push_reports(sh: Shell, reports_dir: Path, date: str) -> bool:
    """Commit and push ``reports/``, ``digests/`` and ``specs/``; returns whether anything was pushed."""
    sh.run("git", "add", "-A", "reports", "digests", "specs", cwd=reports_dir, check=False)
    if sh.ok("git", "diff", "--cached", "--quiet", cwd=reports_dir):
        return False
    sh.run("git", "commit", "-q", "-m", f"provider-watch: {date}", cwd=reports_dir)
    try:
        sh.run("git", "push", cwd=reports_dir)
    except RuntimeError:
        sh.run("git", "pull", "--rebase", "--quiet", cwd=reports_dir)
        sh.run("git", "push", cwd=reports_dir)
    return True


def render_digest(reports_dir: Path, date: str, highlights: Path | None, reports_repo: str) -> Path:
    out = reports_dir / "digests" / f"{date}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = digest.render(
        digest.load_reports(reports_dir, date),
        date=date,
        highlights=highlights.read_text() if highlights else None,
        repo_url=f"https://github.com/{reports_repo}",
    )
    out.write_text(text)
    return out


def worth_an_issue(reports: list[Report]) -> bool:
    """Anything to review, consider, or fix — otherwise the digest is just a record."""
    return any(
        r.meta.get("prs")
        or r.meta.get("error")
        or any(
            isinstance(g, dict) and g.get("action") == "consider" for g in r.meta.get("gaps") or []
        )
        for r in reports
    )


def open_or_update_issue(sh: Shell, reports_repo: str, date: str, body_file: Path) -> str:
    title = f"Provider watch {date}"
    existing = json.loads(
        sh.run(
            "gh",
            "issue",
            "list",
            "--repo",
            reports_repo,
            "--state",
            "all",
            "--search",
            f'"{title}" in:title',
            "--json",
            "number,title,url",
        )
        or "[]"
    )
    match = next((i for i in existing if i["title"] == title), None)
    if match:
        sh.run(
            "gh",
            "issue",
            "edit",
            "--repo",
            reports_repo,
            str(match["number"]),
            "--body-file",
            str(body_file),
        )
        return match["url"]
    return (
        sh.run(
            "gh",
            "issue",
            "create",
            "--repo",
            reports_repo,
            "--title",
            title,
            "--body-file",
            str(body_file),
        )
        .strip()
        .splitlines()[-1]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--date", required=True, help="run date, YYYY-MM-DD")
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS, help="reports checkout")
    parser.add_argument(
        "--repo-root", type=Path, default=REPO_ROOT, help="pipecat checkout holding the branches"
    )
    parser.add_argument("--pipecat-repo", default="pipecat-ai/pipecat")
    parser.add_argument("--reports-repo", default="pipecat-ai/provider-watch")
    parser.add_argument("--pr-cap", type=int, default=DEFAULT_PR_CAP, help="max PRs opened per run")
    parser.add_argument(
        "--finalize", action="store_true", help="also render the digest and open/update the issue"
    )
    parser.add_argument(
        "--highlights", type=Path, help="Markdown inserted at the top of the digest"
    )
    parser.add_argument(
        "--scratch", type=Path, default=Path("/tmp"), help="where temporary worktrees go"
    )
    args = parser.parse_args(argv)

    sh = Shell()
    reports = load_reports(args.reports, args.date)
    outcome = publish_prs(
        reports,
        sh=sh,
        repo_root=args.repo_root,
        pipecat_repo=args.pipecat_repo,
        reports_repo=args.reports_repo,
        date=args.date,
        cap=args.pr_cap,
    )
    outcome.hints_pr = publish_hints(
        reports,
        sh=sh,
        repo_root=args.repo_root,
        pipecat_repo=args.pipecat_repo,
        date=args.date,
        scratch=args.scratch,
    )
    if args.finalize:
        render_digest(args.reports, args.date, args.highlights, args.reports_repo)
    outcome.reports_pushed = push_reports(sh, args.reports, args.date)
    if args.finalize and worth_an_issue(reports):
        outcome.issue_url = open_or_update_issue(
            sh, args.reports_repo, args.date, args.reports / "digests" / f"{args.date}.md"
        )

    print(json.dumps(outcome.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
