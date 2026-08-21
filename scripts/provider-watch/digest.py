#!/usr/bin/env python3
#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Render a run digest from the provider-watch reports written on a given date.

Reads the YAML frontmatter of every ``reports/<provider>/<unit>/<date>.md`` in a
reports checkout and renders one Markdown page grouping units by status, listing
opened PRs, errors and open items, and linking each report. An optional
highlights file is inserted at the top. Run::

    uv run python scripts/provider-watch/digest.py --reports ./_reports --date 2026-08-20 \\
        --highlights highlights.md --out ./_reports/digests/2026-08-20.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

STATUS_ORDER = [
    "prs-opened",
    "prs-withheld",
    "needs-judgement",
    "new-upstream",
    "blocked",
    "error",
    "up-to-date",
]
STATUS_LABELS = {
    "prs-opened": "PRs opened, to review",
    "prs-withheld": "PRs withheld (criteria met; per-run PR cap reached)",
    "needs-judgement": "Changes to consider",
    "new-upstream": "New upstream, no action proposed",
    "blocked": "Blocked",
    "error": "Errors",
    "up-to-date": "Up to date",
}


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
        meta.setdefault("status", "error")
        meta["_path"] = path.relative_to(reports_dir).as_posix()
        found.append(meta)
    return found


def _link(report: dict, repo_url: str | None) -> str:
    path = report["_path"]
    return (
        f"[{report['service']}]({repo_url.rstrip('/')}/blob/main/{path})"
        if repo_url
        else f"`{report['service']}`"
    )


def render(reports: list[dict], *, date: str, highlights: str | None, repo_url: str | None) -> str:
    by_status: dict[str, list[dict]] = {}
    for report in reports:
        by_status.setdefault(str(report.get("status")), []).append(report)

    lines = [f"# Provider watch — {date}", ""]
    if highlights:
        lines += [highlights.strip(), ""]

    counts = ", ".join(
        f"{len(by_status[s])} {s}"
        for s in STATUS_ORDER + sorted(set(by_status) - set(STATUS_ORDER))
        if s in by_status
    )
    lines += [f"**{len(reports)} units researched** — {counts or 'none'}.", ""]

    prs = [(r, pr) for r in reports for pr in (r.get("prs") or [])]
    if prs:
        lines += ["## PRs opened, to review", ""]
        for report, pr in prs:
            url = pr.get("url") if isinstance(pr, dict) else str(pr)
            summary = pr.get("summary", "") if isinstance(pr, dict) else ""
            lines.append(f"- {_link(report, repo_url)} — {url}{' — ' + summary if summary else ''}")
        lines.append("")

    for status in STATUS_ORDER + sorted(set(by_status) - set(STATUS_ORDER)):
        group = by_status.get(status)
        if not group or status == "up-to-date":
            continue
        lines += [f"## {STATUS_LABELS.get(status, status)}", ""]
        for report in group:
            summary = str(report.get("summary") or "").strip()
            lines.append(
                f"- {_link(report, repo_url)} — {report.get('default_model') or '—'}"
                + (f": {summary}" if summary else "")
            )
            for item in report.get("open_items") or []:
                lines.append(f"  - {item}")
        lines.append("")

    up_to_date = by_status.get("up-to-date") or []
    if up_to_date:
        lines += ["## Up to date", "", ", ".join(_link(r, repo_url) for r in up_to_date), ""]

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
