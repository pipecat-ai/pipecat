#!/usr/bin/env python3
#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Publish the provider research produced for one date.

Works entirely from disk — the date's reports under ``_reports`` and the local
``provider-watch/*`` branches they name — so it composes across research runs:
any number of ``/provider-research`` invocations can write for the same date,
and each publish pass picks up whatever is new. Every step is idempotent:
branches already on origin are not pushed again, a branch with an open PR
adopts that PR, reports already pointing at a PR URL are left alone, and the
digest issue is edited rather than duplicated.

For each report whose ``prs`` list has an entry in ``state: branch``:

1. push the branch and open a draft PR (title and body from the branch's
   commit messages — a single commit verbatim, several stitched with the
   report's summary as the title — plus a link to the report);
2. rewrite the report — frontmatter entry to ``state: open`` with the URL,
   and the body's branch/review line to the URL.

A sweep over the open provider-watch PRs then renames every ``+slug``
changelog fragment to its PR's number — the PRs this pass opened, plus any
left misnamed from a previous killed run.

Then commit and push ``_reports``. With ``--finalize`` it also publishes the
digest — the ``digests/<date>.md`` that ``/provider-research-digest`` rendered,
or a highlights-less render made here when none exists — and opens (or
updates) the digest issue on the reports repo when there is anything to show.
Run::

    uv run python scripts/provider-watch/publish.py --date 2026-08-20
    uv run python scripts/provider-watch/publish.py --date 2026-08-20 --finalize
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import digest  # noqa: E402

REPO_ROOT = HERE.parents[1]
DEFAULT_REPORTS = REPO_ROOT / "_reports"
PR_LABEL = "provider-watch"

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


def _main_base(sh: Shell, repo_root: Path) -> str:
    return (
        "origin/main"
        if sh.ok("git", "rev-parse", "--verify", "--quiet", "origin/main", cwd=repo_root)
        else "main"
    )


FRAGMENT_TYPES = {
    "added",
    "changed",
    "deprecated",
    "removed",
    "fixed",
    "security",
    "performance",
    "other",
}


def _fragment_type(name: str) -> str:
    return next((part for part in name.split(".") if part in FRAGMENT_TYPES), "other")


def _rename_changelog_fragments(sh: Shell, repo_root: Path, branch: str, pr_url: str) -> None:
    """Give a PR's changelog fragments the PR's number.

    Researchers write towncrier's ``+slug`` orphan form because no PR exists
    when a branch is committed, and must not guess a number. Once the PR is
    open its number is known: one follow-up commit renames every fragment the
    branch adds that does not already carry it — orphans and wrong guesses
    alike — to ``<number>.<type>.md``, with ``.2``/``.3`` counters when a
    branch adds several of one type. Works from the branch as pushed, in a
    detached worktree, so it needs no local branch and cannot collide with a
    leftover researcher worktree still holding one.
    """
    number = pr_url.rstrip("/").split("/")[-1]
    if not number.isdigit():
        return
    sh.run("git", "fetch", "--quiet", "origin", branch, cwd=repo_root)
    added = sh.run(
        "git",
        "diff",
        "--name-only",
        "--diff-filter=A",
        # Three-dot: only what the branch itself adds, however far main has
        # moved since the branch was cut.
        f"{_main_base(sh, repo_root)}...FETCH_HEAD",
        "--",
        "changelog/",
        cwd=repo_root,
    ).split()
    rename = sorted(p for p in added if not Path(p).name.startswith(f"{number}."))
    if not rename:
        return
    counters: dict[str, int] = {}
    for path in added:
        if Path(path).name.startswith(f"{number}."):
            fragment_type = _fragment_type(Path(path).name)
            counters[fragment_type] = counters.get(fragment_type, 0) + 1
    workdir = tempfile.mkdtemp(prefix="pw-fragments-")
    try:
        sh.run(
            "git", "worktree", "add", "--quiet", "--detach", workdir, "FETCH_HEAD", cwd=repo_root
        )
        for path in rename:
            fragment_type = _fragment_type(Path(path).name)
            counters[fragment_type] = counters.get(fragment_type, 0) + 1
            counter = counters[fragment_type]
            suffix = "" if counter == 1 else f".{counter}"
            sh.run(
                "git",
                "mv",
                path,
                f"changelog/{number}.{fragment_type}{suffix}.md",
                cwd=Path(workdir),
            )
        sh.run(
            "git",
            "commit",
            "-q",
            "-m",
            f"Name the changelog fragments after PR #{number}",
            cwd=Path(workdir),
        )
        sh.run("git", "push", "origin", f"HEAD:refs/heads/{branch}", cwd=Path(workdir))
    finally:
        sh.run("git", "worktree", "remove", "--force", workdir, cwd=repo_root, check=False)


def rename_open_pr_fragments(sh: Shell, repo_root: Path, pipecat_repo: str) -> list[str]:
    """Rename the ``+slug`` fragments on every open provider-watch PR to its number.

    This sweep is the only place fragments are named: freshly opened PRs still
    carry their researchers' ``+slug`` fragments, and a killed run's publish
    can leave PRs misnamed with no local branch surviving — so naming works
    from the PRs themselves, whatever run opened them. A correctly named PR
    costs one lookup. Returns the failures, as skip messages.
    """
    skipped: list[str] = []
    owner = pipecat_repo.split("/")[0]
    prs = json.loads(
        sh.run(
            "gh",
            "pr",
            "list",
            "--repo",
            pipecat_repo,
            "--label",
            PR_LABEL,
            "--state",
            "open",
            "--json",
            "number,headRefName,headRepositoryOwner",
        )
        or "[]"
    )
    for pr in prs:
        number = str(pr.get("number") or "")
        head = pr.get("headRefName") or ""
        head_owner = (pr.get("headRepositoryOwner") or {}).get("login", "")
        # Only branches the bot owns; never push to a fork or a human's branch.
        if not head.startswith("provider-watch/") or head_owner != owner:
            continue
        files = json.loads(
            sh.run("gh", "pr", "view", number, "--repo", pipecat_repo, "--json", "files") or "{}"
        ).get("files")
        fragments = [f["path"] for f in files or [] if f["path"].startswith("changelog/")]
        if all(Path(p).name.startswith(f"{number}.") for p in fragments):
            continue
        try:
            _rename_changelog_fragments(
                sh, repo_root, head, f"https://github.com/{pipecat_repo}/pull/{number}"
            )
        except RuntimeError as exc:
            skipped.append(f"{head}: fragments not renamed: {exc}")
    return skipped


def _pr_title_body(sh: Shell, repo_root: Path, branch: str, summary: str) -> tuple[str, str]:
    """PR title and body from the branch's commits (one commit per item).

    A single commit becomes the PR verbatim; several are stitched into one body,
    one section per commit oldest-first, titled by the report's summary for the
    branch.
    """
    base = _main_base(sh, repo_root)
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
) -> Outcome:
    """Open PRs for branch-state entries and rewrite the reports."""
    outcome = Outcome()

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
                outcome.opened.append(url)

            pr.update({"state": "open", "url": url, "opened": date})
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


def ensure_digest(reports_dir: Path, date: str, reports_repo: str) -> Path:
    """The digest to publish: the one ``/provider-research-digest`` rendered, untouched,
    else a highlights-less render so ``--finalize`` still has a digest to publish."""
    out = reports_dir / "digests" / f"{date}.md"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    text = digest.render(
        digest.load_reports(reports_dir, date),
        date=date,
        highlights=None,
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
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="also publish the digest (digests/<date>.md, rendered by "
        "/provider-research-digest; a highlights-less one is rendered here if "
        "missing) and open/update the digest issue",
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
    )
    outcome.skipped += rename_open_pr_fragments(sh, args.repo_root, args.pipecat_repo)
    if args.finalize:
        digest_file = ensure_digest(args.reports, args.date, args.reports_repo)
    outcome.reports_pushed = push_reports(sh, args.reports, args.date)
    if args.finalize and worth_an_issue(reports):
        outcome.issue_url = open_or_update_issue(sh, args.reports_repo, args.date, digest_file)

    print(json.dumps(outcome.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
