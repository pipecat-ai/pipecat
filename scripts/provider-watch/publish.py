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
   commit messages — a single commit verbatim, several stitched with the
   report's summary as the title — plus a link to the report), subject to
   the per-run cap;
2. rewrite the report — frontmatter entry to ``state: open`` with the URL,
   and the body's branch/review line to the URL.

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
PR_LABEL = "provider-watch"
DEFAULT_PR_CAP = 8

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


def _pr_title_body(sh: Shell, repo_root: Path, branch: str, summary: str) -> tuple[str, str]:
    """PR title and body from the branch's commits (one commit per item).

    A single commit becomes the PR verbatim; several are stitched into one body,
    one section per commit oldest-first, titled by the report's summary for the
    branch.
    """
    base = (
        "origin/main"
        if sh.ok("git", "rev-parse", "--verify", "--quiet", "origin/main", cwd=repo_root)
        else "main"
    )
    hashes = sh.run("git", "rev-list", "--reverse", f"{base}..{branch}", cwd=repo_root).split()
    if not hashes:
        hashes = [branch]
    commits = [
        (
            sh.run("git", "log", "-1", "--format=%s", h, cwd=repo_root).strip(),
            sh.run("git", "log", "-1", "--format=%b", h, cwd=repo_root).strip(),
        )
        for h in hashes
    ]
    if len(commits) == 1:
        return commits[0]
    body = "\n\n".join(f"## {s}\n\n{b}".rstrip() for s, b in commits)
    return summary.strip() or commits[-1][0], body


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
                subject, body = _pr_title_body(sh, repo_root, branch, str(pr.get("summary") or ""))
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


def push_reports(sh: Shell, reports_dir: Path, date: str) -> bool:
    """Commit and push ``reports/`` and ``digests/``; returns whether anything was pushed."""
    present = [d for d in ("reports", "digests") if (reports_dir / d).is_dir()]
    if not present:
        return False
    sh.run("git", "add", "-A", *present, cwd=reports_dir, check=False)
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
    parser.add_argument("--reports-repo", default="pipecat-ai/provider-watch-reports")
    parser.add_argument("--pr-cap", type=int, default=DEFAULT_PR_CAP, help="max PRs opened per run")
    parser.add_argument(
        "--finalize", action="store_true", help="also render the digest and open/update the issue"
    )
    parser.add_argument(
        "--highlights", type=Path, help="Markdown inserted at the top of the digest"
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
