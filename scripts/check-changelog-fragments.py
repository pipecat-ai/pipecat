#!/usr/bin/env python3
"""Check that every changelog fragment is named so towncrier picks it up.

Usage::

    python scripts/check-changelog-fragments.py

A fragment is named ``<PR number>.<type>.md``, or ``<PR number>.<type>.<n>.md``
for a second entry of the same type. The valid types and the fragments
directory are read from ``[tool.towncrier]`` in ``pyproject.toml``, so adding a
type there is the only change needed. The template that config names is
skipped, as are dotfiles.

towncrier's ``+slug`` orphan form is rejected: its entry would carry no PR link.
This runs on every PR (``format.yaml``) and again before a release
(``generate-changelog.yml``). Exits 1 and lists the offending files if any are
misnamed. An empty directory passes.
"""

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    """List misnamed changelog fragments and return the exit code."""
    with open(ROOT / "pyproject.toml", "rb") as f:
        config = tomllib.load(f)["tool"]["towncrier"]

    types = [t["directory"] for t in config["type"]]
    directory = ROOT / config["directory"]
    template = (ROOT / config["template"]).resolve()
    pattern = re.compile(rf"[0-9]+\.({'|'.join(map(re.escape, types))})(\.[0-9]+)?\.md")

    invalid = sorted(
        path.name
        for path in directory.iterdir()
        if path.resolve() != template
        and not path.name.startswith(".")
        and not pattern.fullmatch(path.name)
    )

    if invalid:
        print("Error: badly named changelog fragments:")
        for name in invalid:
            print(f"  - {name}")
        print("Name them <PR number>.<type>.md or <PR number>.<type>.<n>.md,")
        print(f"with type one of: {' '.join(types)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
