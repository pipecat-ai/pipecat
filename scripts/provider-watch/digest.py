#!/usr/bin/env python3
#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Render a run digest from the provider-watch reports written on a given date.

Reads the YAML frontmatter of every ``reports/<provider>/<unit>/<date>.md`` in a
reports checkout and renders one Markdown page: PRs to review, branches awaiting
a PR, changes to consider (with how long each gap has been open), units that
could not be researched, and units with nothing new — each linking its report.
An optional highlights file is inserted at the top. Run::

    uv run python scripts/provider-watch/digest.py --reports ./_reports --date 2026-08-20 \\
        --highlights highlights.md --out ./_reports/digests/2026-08-20.md
"""

from __future__ import annotations

import argparse
import sys
from datetime import date as _date
from pathlib import Path

import yaml

PRIORITY_ORDER = ["high", "medium", "low"]


def parse_frontmatter(text: str) -> dict:
    """Return the YAML frontmatter of a report, or an empty dict."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    data = yaml.safe_load(text[3:end]) or {}
    return data if isinstance(data, dict) else {}


def load_reports(reports_dir: Path, date: str) -> list[dict]:
    """Frontmatter of every report for ``date``, each tagged with its relative path."""
    found = []
    for path in sorted(reports_dir.glob(f"reports/*/*/{date}.md")):
        meta = parse_frontmatter(path.read_text())
        meta.setdefault("service", "/".join(path.parts[-3:-1]))
        if not meta.get("error") and not path.read_text().startswith("---"):
            meta["error"] = "report has no frontmatter"
        meta["_path"] = path.relative_to(reports_dir).as_posix()
        found.append(meta)
    return found


def _summary(pr: dict) -> str:
    summary = str(pr.get("summary") or "").strip()
    return f" — {summary}" if summary else ""


def _link(report: dict, repo_url: str | None) -> str:
    path = report["_path"]
    return (
        f"[{report['service']}]({repo_url.rstrip('/')}/blob/main/{path})"
        if repo_url
        else f"`{report['service']}`"
    )


def _age(first_seen, date: str) -> str:
    """``(since 2026-08-06, 2 weeks)`` for a gap first seen before this run."""
    try:
        seen = _date.fromisoformat(str(first_seen))
        days = (_date.fromisoformat(date) - seen).days
    except (TypeError, ValueError):
        return ""
    if days < 7:
        return ""
    weeks = days // 7
    return f" (since {seen}, {weeks} week{'s' if weeks != 1 else ''})"


def render(reports: list[dict], *, date: str, highlights: str | None, repo_url: str | None) -> str:
    prs = [(r, pr) for r in reports for pr in (r.get("prs") or []) if isinstance(pr, dict)]
    open_prs = [(r, pr) for r, pr in prs if pr.get("state") in {"open", "merged", "closed"}]
    branches = [(r, pr) for r, pr in prs if pr.get("state") == "branch"]
    considerations = [
        (r, gap)
        for r in reports
        for gap in (r.get("gaps") or [])
        if isinstance(gap, dict) and gap.get("action") == "consider"
    ]
    errors = [r for r in reports if r.get("error")]
    quiet = [
        r
        for r in reports
        if not r.get("prs")
        and not r.get("error")
        and not any(
            isinstance(g, dict) and g.get("action") == "consider" for g in r.get("gaps") or []
        )
    ]

    lines = [f"# Provider watch — {date}", ""]
    if highlights:
        lines += [highlights.strip(), ""]
    lines += [
        f"**{len(reports)} units researched** — {len(open_prs)} PRs, {len(branches)} branches, "
        f"{len(considerations)} changes to consider, {len(errors)} errors, {len(quiet)} with nothing new.",
        "",
    ]

    if open_prs:
        lines += ["## PRs to review", ""]
        for report, pr in open_prs:
            state = f" ({pr['state']})" if pr.get("state") != "open" else ""
            lines.append(f"- {_link(report, repo_url)} — {pr.get('url')}{state}{_summary(pr)}")
        lines.append("")
    if branches:
        capped = [b for b in branches if b[1].get("capped")]
        title = "## Branches not opened as PRs" + (
            " (per-run cap reached)" if capped else " (dry run)"
        )
        lines += [title, ""]
        for report, pr in branches:
            branch = pr.get("branch")
            lines.append(
                f"- {_link(report, repo_url)} — `{branch}` — review: `git show {branch}`{_summary(pr)}"
            )
        lines.append("")
    if considerations:
        lines += ["## Changes to consider", ""]
        by_priority: dict[str, list] = {}
        for report, gap in considerations:
            by_priority.setdefault(str(gap.get("priority") or "unranked"), []).append((report, gap))
        for priority in PRIORITY_ORDER + sorted(set(by_priority) - set(PRIORITY_ORDER)):
            group = by_priority.get(priority)
            if not group:
                continue
            items = [
                f"- {_link(report, repo_url)} — {gap.get('item')}{_age(gap.get('first_seen'), date)}"
                + (f" — {gap['note']}" if gap.get("note") else "")
                for report, gap in group
            ]
            if priority == "low":
                lines += (
                    [f"<details><summary><b>Low</b> ({len(items)})</summary>", ""]
                    + items
                    + ["", "</details>", ""]
                )
            else:
                lines += [f"**{priority.capitalize()}**", ""] + items + [""]
    if errors:
        lines += ["## Did not complete", ""]
        lines += [f"- {_link(r, repo_url)} — {r['error']}" for r in errors]
        lines.append("")
    if quiet:
        lines += ["## Nothing new", "", ", ".join(_link(r, repo_url) for r in quiet), ""]

    lines += [
        "---",
        "To record a decision about an item above, reply on this issue naming the unit and enough of "
        "the item to identify it, one decision per line — e.g. `deepgram/stt, diarize_model: skip, "
        "the extra= workaround is fine` or `openai/realtime, tool_choice: done in #5400`. "
        "The next run reads these comments.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--reports", required=True, type=Path, help="reports repo checkout")
    parser.add_argument("--date", required=True, help="run date, YYYY-MM-DD")
    parser.add_argument("--highlights", type=Path, help="Markdown inserted under the title")
    parser.add_argument(
        "--repo-url",
        default="https://github.com/pipecat-ai/provider-watch",
        help="link base; empty for plain names",
    )
    parser.add_argument("--out", type=Path, help="write here instead of stdout")
    args = parser.parse_args(argv)

    reports = load_reports(args.reports, args.date)
    text = render(
        reports,
        date=args.date,
        highlights=args.highlights.read_text() if args.highlights else None,
        repo_url=args.repo_url or None,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
