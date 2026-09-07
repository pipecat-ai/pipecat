#!/usr/bin/env python3
"""Check the licenses of pipecat's core dependencies against an allowlist.

Pipecat is BSD 2-Clause licensed; its core install (``pip install pipecat-ai``)
must not pull in any package whose license would impose additional terms on
applications built with it — copyleft licenses like the GPL in particular.
Optional extras are deliberately out of scope: they are opt-in, and their terms
are set by the service vendors.

The check runs against the *resolved* core dependency set (direct and
transitive, all platforms and Python versions) exported from ``uv.lock``, so it
also catches a dependency that changes license in a version bump within an
already-allowed range. License metadata is fetched from PyPI for the exact
locked versions.

Usage::

    python3 scripts/check_licenses.py

Requires ``uv`` on PATH and network access to pypi.org. Exits non-zero when a
package has a denied license or no recognizable license metadata at all — add
such packages to ``KNOWN_PACKAGES`` after a manual review.
"""

import json
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# Licenses acceptable in the core dependency set. Permissive licenses, plus
# weak copyleft (LGPL, MPL) whose obligations stay contained to the library
# itself when used unmodified through its public interface.
ALLOWED_LICENSES = [
    r"\bmit\b",
    r"\bbsd\b",
    r"\bapache\b",
    r"\bisc\b",
    r"\bpsf\b",
    r"python software foundation",
    r"\blgpl\b",
    r"lesser general public license",
    r"\bmpl\b",
    r"mozilla public license",
    r"\bzlib\b",
    r"\bcc0\b",
    r"\b0bsd\b",
    r"\bunlicense\b",
    r"public domain",
    r"\bhpnd\b",
    r"\bbsl-1\.0\b",
    r"\bboost\b",
]

# Packages whose PyPI metadata is missing or unrecognizable but whose license
# was verified manually. Map of package name -> justification (with the actual
# license and where it was verified).
KNOWN_PACKAGES: dict[str, str] = {}

# LGPL phrasings are removed from the metadata text before scanning for GPL, so
# these patterns only match the strong-copyleft licenses.
_LGPL_RE = re.compile(r"(?:library or )?lesser general public license|\blgpl[v0-9.+-]*\b")
_DENIED_RE = re.compile(r"\bgpl\b|\bgpl[v0-9.+-]+\b|general public license|\baffero\b|\bagpl\b")

_ALLOWED_RE = [re.compile(p) for p in ALLOWED_LICENSES]


def core_dependencies() -> list[tuple[str, str]]:
    """Return the resolved (name, version) core dependency set from uv.lock."""
    out = subprocess.run(
        ["uv", "export", "--no-dev", "--no-emit-project", "--no-hashes", "--frozen"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    deps = []
    for line in out.splitlines():
        line = line.split(";")[0].strip()
        if not line or line.startswith(("#", "-")) or "==" not in line:
            continue
        name, version = line.split("==")
        deps.append((re.sub(r"\[.*\]", "", name).strip(), version.strip()))
    return sorted(set(deps))


def license_metadata(name: str, version: str) -> list[str]:
    """Fetch a package version's license metadata from PyPI.

    Returns the metadata sources in decreasing order of authority: the PEP 639
    SPDX license expression, then the trove classifiers, then the free-text
    ``License`` field (which some packages fill with an entire license text).
    Each is a lowercase string; absent sources are empty.
    """
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                info = json.load(response)["info"]
            classifiers = [c for c in info.get("classifiers") or [] if c.startswith("License ::")]
            return [
                (info.get("license_expression") or "").lower(),
                " ".join(classifiers).lower(),
                (info.get("license") or "").lower(),
            ]
        except Exception as e:  # noqa: BLE001 — retry any fetch/parse hiccup
            last_error = e
            time.sleep(2**attempt)
    raise RuntimeError(f"could not fetch license metadata for {name}=={version}: {last_error}")


def check(name: str, version: str) -> str | None:
    """Return a violation message for the package, or None if it passes.

    The most authoritative metadata source that yields a verdict wins; an
    inconclusive source (present but matching neither list) falls through to
    the next one.
    """
    if name in KNOWN_PACKAGES:
        return None
    sources = license_metadata(name, version)
    for source in sources:
        if not source:
            continue
        if _DENIED_RE.search(_LGPL_RE.sub("", source)):
            return f"{name}=={version}: denied license: {source[:100]!r}"
        if any(p.search(source) for p in _ALLOWED_RE):
            return None
    if not any(sources):
        return f"{name}=={version}: no license metadata on PyPI"
    return f"{name}=={version}: unrecognized license metadata: {' '.join(sources)[:100]!r}"


def main() -> int:
    deps = core_dependencies()
    print(f"Checking licenses of {len(deps)} resolved core dependencies...")
    with ThreadPoolExecutor(max_workers=12) as executor:
        violations = [v for v in executor.map(lambda d: check(*d), deps) if v]
    if violations:
        print(f"\n{len(violations)} core dependency license violation(s):\n", file=sys.stderr)
        for violation in sorted(violations):
            print(f"  {violation}", file=sys.stderr)
        print(
            "\nCore dependencies must be permissively licensed (or weak copyleft:"
            " LGPL/MPL). If a package's metadata is wrong or missing, verify its"
            " actual license manually and add it to KNOWN_PACKAGES in"
            " scripts/check_licenses.py with a justification.",
            file=sys.stderr,
        )
        return 1
    print("OK: all core dependency licenses are allowed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
