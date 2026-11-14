#!/usr/bin/env python3
#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Generate the removal history (``scripts/deprecations/removals.json``).

Records symbols that were deprecated in a prior release and have since been
removed, so downstream tooling — the pipecat-context-hub ``check_deprecation``
query, IDE plugins, docs — can still answer "``X`` was removed in ``V``, use
``Y``" after the symbol (and its ``.. deprecated::`` marker) is gone from the
source. ``deprecations.json`` stays a pure snapshot of the current tree; this is
its additive, cumulative companion.

Removals are detected by diffing the **previous released tag's**
``deprecations.json`` against the one in the current working tree: any subject
that was present then and is gone now was removed in the release being prepared.
Run during release-prep, **before the tag is cut**, with the release version::

    uv run python scripts/deprecations/generate_removals.py --version 2.0.0

Symbols moved or removed *without* a prior deprecation (e.g. alpha features) can't be
detected this way. Record them in ``relocations.json`` (hand-maintained); they are merged
into ``removals.json`` here. After writing, the script warns about any public module that
existed in the previous release, is gone now, and has no record, so a silent move is
caught at release-prep. ``--previous-tag`` overrides the tag diffed against.

The file is cumulative — prior entries are never rewritten — and fully
reconstructable from the tag history, so it is never primary data. The ``--version``
value is the release being prepared (the workflow supplies it); the previous tag
is discovered automatically.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = Path(__file__).resolve().parent / "deprecations.json"
REMOVALS_PATH = Path(__file__).resolve().parent / "removals.json"
# Hand-maintained moves/removals that were never deprecated first (see load_relocations).
RELOCATIONS_PATH = Path(__file__).resolve().parent / "relocations.json"
SOURCE_ROOT = REPO_ROOT / "src" / "pipecat"
# Repo-relative, POSIX-style, for `git ls-tree` against a tag.
SOURCE_REL = "src/pipecat"
# Path used with `git show <tag>:<path>` — must be repo-relative and POSIX-style.
REGISTRY_REL = "scripts/deprecations/deprecations.json"
SCHEMA_VERSION = 1

_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def _version_key(tag: str) -> tuple[int, int, int] | None:
    """Sort key for a ``vX.Y.Z`` tag, or ``None`` if it isn't a release tag."""
    match = _VERSION_RE.match(tag)
    return tuple(int(g) for g in match.groups()) if match else None  # type: ignore[return-value]


def subjects_of(registry: dict) -> dict[str, dict]:
    """Map ``subject`` → record for a ``deprecations.json`` document."""
    return {r["subject"]: r for r in registry.get("deprecations", []) if r.get("subject")}


_REMOVAL_FIELDS = (
    "subject",
    "module",
    "kind",
    "deprecated_in",
    "removed_in",
    "announced_removed_in",
    "relation",
    "replacement",
    "message",
)


def load_relocations(path: Path = RELOCATIONS_PATH) -> list[dict]:
    """Hand-maintained removal records for symbols that were never deprecated first.

    ``compute_removals`` only sees subjects that were deprecated in the previous
    release, so an alpha feature moved or deleted outright (no deprecation period)
    is invisible to it. Record those in ``relocations.json``; they are merged into
    ``removals.json`` so downstream tooling reads a single removal history.

    Returns ``[]`` if the file doesn't exist. Raises ``ValueError`` on a malformed
    record, so a typo fails release-prep instead of silently dropping the entry.
    """
    if not path.exists():
        return []
    records = json.loads(path.read_text(encoding="utf-8")).get("relocations", [])
    seen: set[str] = set()
    for rec in records:
        subject = rec.get("subject")
        if not subject or subject in seen:
            raise ValueError(f"{path.name}: missing or duplicate subject {subject!r}")
        seen.add(subject)
        if set(rec) != set(_REMOVAL_FIELDS):
            raise ValueError(
                f"{path.name}: {subject!r} must have exactly the fields {_REMOVAL_FIELDS}"
            )
        if not _VERSION_RE.match(rec["removed_in"] or ""):
            raise ValueError(f"{path.name}: {subject!r} removed_in must be X.Y.Z")
        if not rec["replacement"] or not rec["message"]:
            raise ValueError(f"{path.name}: {subject!r} needs a replacement and a message")
    return records


def compute_removals(
    prev_registry: dict | None,
    current_registry: dict,
    version: str,
    existing: list[dict],
    relocations: list[dict] | None = None,
) -> list[dict]:
    """Return the updated, cumulative list of removal records.

    A subject present in ``prev_registry`` and absent from ``current_registry`` was
    removed in ``version``. Its fields are carried forward from the previous
    deprecation record (its ``location`` is intentionally dropped — it points at
    source that no longer exists). Already-recorded subjects are left untouched, so
    repeated runs are idempotent.

    A subject the current registry still records members of (``X`` when
    ``X.member`` is present) was re-recorded at a finer granularity, not removed.

    ``relocations`` are hand-maintained records for symbols that were never
    deprecated (see ``load_relocations``). They are added like detected removals,
    skipping any subject already recorded, so repeated runs stay idempotent.
    """
    prev = subjects_of(prev_registry) if prev_registry else {}
    current = set(subjects_of(current_registry))
    known = {r["subject"] for r in existing}

    removals = list(existing)
    for subject, rec in prev.items():
        if subject in current or subject in known:
            continue
        if any(s.startswith(f"{subject}.") for s in current):
            continue
        removals.append(
            {
                "subject": subject,
                "module": rec.get("module"),
                "kind": rec.get("kind"),
                "deprecated_in": rec.get("deprecated_in"),
                "removed_in": version,
                "announced_removed_in": rec.get("removed_in"),
                "relation": rec.get("relation"),
                "replacement": rec.get("replacement"),
                "message": rec.get("message"),
            }
        )
    recorded = {r["subject"] for r in removals}
    for rec in relocations or []:
        if rec["subject"] not in recorded:
            removals.append({k: rec[k] for k in _REMOVAL_FIELDS})
            recorded.add(rec["subject"])
    # Group by the release a symbol was removed in, then by name — deterministic
    # output for clean diffs, regardless of dict iteration order.
    removals.sort(key=lambda r: (r.get("removed_in") or "", r["subject"]))
    return removals


def build_document(removals: list[dict]) -> dict:
    """Wrap removal records in the registry document envelope."""
    return {
        "_comment": (
            "AUTO-GENERATED by scripts/deprecations/generate_removals.py at release-prep. "
            "Cumulative and reconstructable from tag history plus relocations.json. DO NOT EDIT BY HAND; "
            "record never-deprecated moves in relocations.json instead."
        ),
        "schema_version": SCHEMA_VERSION,
        "removals": removals,
    }


def render(document: dict) -> str:
    """Deterministic JSON text (stable across runs for clean git diffs)."""
    return json.dumps(document, indent=2, sort_keys=False, ensure_ascii=False) + "\n"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


def previous_release_tag() -> str | None:
    """The highest ``vX.Y.Z`` tag whose tree carries ``deprecations.json``.

    At release-prep the version being released is not tagged yet, so the highest
    existing release tag is the previous release. Tags without the registry
    (pre-#4726) are skipped, which is what makes the first registry-bearing
    release an empty baseline.
    """
    listed = _git("tag", "--list", "v*")
    best: str | None = None
    best_key: tuple[int, int, int] | None = None
    for tag in listed.stdout.split():
        key = _version_key(tag)
        if key is None:
            continue
        if _git("cat-file", "-e", f"{tag}:{REGISTRY_REL}").returncode != 0:
            continue
        if best_key is None or key > best_key:
            best, best_key = tag, key
    return best


def registry_at_tag(tag: str) -> dict | None:
    """Load ``deprecations.json`` as it stood at ``tag``, or ``None`` if absent."""
    proc = _git("show", f"{tag}:{REGISTRY_REL}")
    if proc.returncode != 0:
        return None
    return json.loads(proc.stdout)


def load_existing_removals() -> list[dict]:
    """Existing removal records, or ``[]`` if the file doesn't exist yet."""
    if not REMOVALS_PATH.exists():
        return []
    return json.loads(REMOVALS_PATH.read_text(encoding="utf-8")).get("removals", [])


def _module_name(path: str) -> str | None:
    """Dotted module for a ``src/pipecat/...py`` path, or ``None`` if it is private."""
    parts = path.removeprefix("src/").removesuffix(".py").split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts or any(p.startswith("_") for p in parts):
        return None
    return ".".join(parts)


def modules_at_tag(tag: str) -> set[str]:
    """Public module paths under ``src/pipecat`` as they stood at ``tag``."""
    proc = _git("ls-tree", "-r", "--name-only", tag, SOURCE_REL)
    names = (_module_name(p) for p in proc.stdout.split() if p.endswith(".py"))
    return {n for n in names if n}


def modules_in_tree() -> set[str]:
    """Public module paths under ``src/pipecat`` in the working tree."""
    names = (_module_name(p.relative_to(REPO_ROOT).as_posix()) for p in SOURCE_ROOT.rglob("*.py"))
    return {n for n in names if n}


def unrecorded_vanished_modules(
    prev_modules: set[str], current_modules: set[str], removals: list[dict]
) -> list[str]:
    """Public modules present in the previous release, gone now, with no record.

    ``compute_removals`` only sees symbols that were deprecated first, so a module
    deleted or moved outright is otherwise silent. A vanished module is accounted
    for when a removal record names it, or a package above it, as its ``module`` or
    (for ``kind == "module"``) its ``subject``.
    """
    recorded = {r["module"] for r in removals if r.get("module")}
    recorded |= {r["subject"] for r in removals if r.get("kind") == "module"}
    return sorted(
        m
        for m in prev_modules - current_modules
        if not any(m == r or m.startswith(f"{r}.") for r in recorded)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        required=True,
        help="Release version being prepared (e.g. 2.0.0). Stamped as removed_in.",
    )
    parser.add_argument(
        "--previous-tag",
        help="Release tag to diff against (default: the highest vX.Y.Z tag with a registry).",
    )
    args = parser.parse_args()

    version = args.version.lstrip("v")
    if not _VERSION_RE.match(version):
        print(f"--version must be X.Y.Z, got {args.version!r}", file=sys.stderr)
        return 1

    if not REGISTRY_PATH.exists():
        print(
            f"{REGISTRY_PATH.name} not found — run generate.py first.",
            file=sys.stderr,
        )
        return 1
    current_registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    prev_tag = args.previous_tag or previous_release_tag()
    prev_registry = registry_at_tag(prev_tag) if prev_tag else None
    existing = load_existing_removals()

    try:
        relocations = load_relocations()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    removals = compute_removals(prev_registry, current_registry, version, existing, relocations)
    REMOVALS_PATH.write_text(render(build_document(removals)), encoding="utf-8")

    new_count = len(removals) - len(existing)
    if prev_registry is None:
        print(
            "No previous release tag carries deprecations.json — baseline release "
            "(no removals can be detected yet)."
        )
    print(
        f"Wrote {len(removals)} removal record(s) "
        f"({new_count} new in {version}; previous tag: {prev_tag}) "
        f"to {REMOVALS_PATH.relative_to(REPO_ROOT)}"
    )

    if prev_tag:
        vanished = unrecorded_vanished_modules(
            modules_at_tag(prev_tag), modules_in_tree(), removals
        )
        if vanished:
            print(
                f"WARNING: {len(vanished)} public module(s) present in {prev_tag} are gone "
                "with no removal record. Downstream tooling will report them as current.\n"
                "Record each in scripts/deprecations/relocations.json (moved or removed "
                "without a deprecation), or restore it:\n  " + "\n  ".join(vanished),
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
