#!/usr/bin/env python3
"""Generate release notes from CHANGELOG.md for use in GitHub Releases (or anywhere).

Usage::

    python scripts/release-changelog.py VERSION
    python scripts/release-changelog.py 1.1.0
    python scripts/release-changelog.py --file path/to/CHANGELOG.md 1.1.0

Extracts the requested ``## [VERSION]`` section (heading and ``### …``
subheadings included) and prints it to stdout. The only transformation
applied is collapsing each entry paragraph onto a single line, so it is
suitable for pasting into release notes that don't need 80-column wrapping.
``(PR [...])`` references stay on their own two-space-indented continuation
line. The input file is never modified.

Every paragraph is unfilled. Indentation is preserved — each logical line
(bullets, sub-bullets, and the ``(PR [...])`` continuation) keeps its
leading whitespace; only the wrapped continuation lines that follow them
get joined back. Code-block paragraphs (triple-backtick fences, or deeply
indented blocks with no list markers around) are passed through untouched.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"

PARAGRAPH_SPLIT_RE = re.compile(r"(\n[ \t]*\n+)")
CONTINUATION_RE = re.compile(r"\n(?![ \t]*[-*+] )[ \t]+")
PR_RE = re.compile(r" \(PR \[")
CODE_FENCE_RE = re.compile(r"^\s*```")
DEEP_INDENT_RE = re.compile(r"^ {4,}")
LIST_MARKER_RE = re.compile(r"^[ \t]*[-*+] ")
SECTION_HEADING_RE = re.compile(r"^## \[", re.MULTILINE)


def is_code_block_paragraph(paragraph: str) -> bool:
    lines = paragraph.splitlines()
    if any(CODE_FENCE_RE.match(line) for line in lines):
        return True
    # In a list paragraph, 4-space indent is a sub-bullet continuation, not
    # code. Only treat deep-indented lines as code when no bullets are around
    # to claim them.
    if any(LIST_MARKER_RE.match(line) for line in lines):
        return False
    return any(DEEP_INDENT_RE.match(line) for line in lines)


def unfill_entry(paragraph: str) -> str:
    joined = CONTINUATION_RE.sub(" ", paragraph)
    return PR_RE.sub("\n  (PR [", joined, count=1)


def process(text: str) -> str:
    parts = PARAGRAPH_SPLIT_RE.split(text)
    for i in range(0, len(parts), 2):
        para = parts[i]
        if para and not is_code_block_paragraph(para):
            parts[i] = unfill_entry(para)
    return "".join(parts)


def extract_section(text: str, version: str) -> str:
    head_re = re.compile(rf"^## \[{re.escape(version)}\]", re.MULTILINE)
    m = head_re.search(text)
    if not m:
        sys.exit(f"error: version {version!r} not found")
    nxt = SECTION_HEADING_RE.search(text, m.end())
    end = nxt.start() if nxt else len(text)
    return text[m.start() : end]


def main() -> None:
    summary = (__doc__ or "").splitlines()[0]
    parser = argparse.ArgumentParser(description=summary)
    parser.add_argument("version", help="Version to extract (e.g. 1.1.0).")
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_CHANGELOG,
        help=f"Path to the changelog (default: {DEFAULT_CHANGELOG}).",
    )
    args = parser.parse_args()

    text = args.file.read_text()
    section = extract_section(text, args.version)
    sys.stdout.write(process(section))


if __name__ == "__main__":
    main()
