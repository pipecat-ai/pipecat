#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Audit: text file IO under ``src/pipecat`` says which encoding it uses.

``open()``, ``Path.read_text()`` and ``Path.write_text()`` fall back to
``locale.getencoding()`` when no encoding is passed: UTF-8 on Linux and macOS, the
ANSI code page on Windows. Anything that reads or writes a file containing a
typographic quote, an accent or a CJK character then fails to decode — or writes a
file that only the machine that produced it can read.

That is not theoretical: the eval manifest, the eval artifacts, the Tone model
config and the CLI project config each had to be fixed for exactly this, and each
one was only found by running the suite on Windows.

The check is an AST pass rather than a regex so that binary modes, ``io.open``
aliases and attribute calls like ``wave.open`` or ``Image.open`` are not confused
with text IO.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).parent.parent / "src" / "pipecat"

# Files allowed to omit an encoding, with the reason. Keep this list short: every
# entry is a place where a Windows-only decode bug can come back.
#
# ``cli/commands/init.py`` talks to the context hub process two ways, and in both the
# encoding belongs to the child: it relays the child's stdout/stderr through an
# inherited descriptor (the log sink), and it captures the child's output of
# ``install`` with ``text=True``. Pinning either side alone would desynchronise the
# two; the fix is to make the hub write UTF-8 and say so on both sides, which is a
# change in that process rather than an ``encoding=`` argument here.
ALLOWED: dict[str, str] = {
    "cli/commands/init.py": (
        "every text call here is child-process IO: the log sink relays a dup'd "
        "descriptor, the failure tail is read back from that same log, and the "
        "install call captures the child's output — the writer's encoding is the "
        "child's."
    ),
}


def _text_calls_without_encoding(tree: ast.AST) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        attr = getattr(func, "attr", None)
        name = getattr(func, "id", None)
        if attr in ("read_text", "write_text"):
            label = f"{attr}()"
        elif name == "open" and isinstance(func, ast.Name):
            mode = None
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = kw.value.value
            if isinstance(mode, str) and "b" in mode:
                continue
            label = "open()"
        elif attr == "open" and isinstance(func.value, ast.Name) and func.value.id == "io":
            mode = None
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = kw.value.value
            if isinstance(mode, str) and "b" in mode:
                continue
            label = "io.open()"
        elif attr in ("run", "check_output", "Popen", "call", "check_call"):
            # subprocess decodes the child's output with the locale too
            textual = any(
                kw.arg in ("text", "universal_newlines")
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in node.keywords
            )
            if not textual or any(kw.arg == "encoding" for kw in node.keywords):
                continue
            label = f"subprocess.{attr}(text=True)"
        else:
            continue
        if any(kw.arg == "encoding" for kw in node.keywords):
            continue
        # read_text may pass the encoding positionally, which is still explicit
        if attr == "read_text" and node.args:
            continue
        found.append((node.lineno, label))
    return found


def test_text_io_states_its_encoding():
    """Every text-mode file call in ``src/pipecat`` passes ``encoding=``."""
    offences = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        rel = path.relative_to(SRC_ROOT).as_posix()
        if rel in ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, label in _text_calls_without_encoding(tree):
            offences.append(f"pipecat/{rel}:{lineno}: {label} without encoding=")

    assert not offences, (
        "Text file IO must state its encoding — Windows otherwise decodes with the "
        "ANSI code page:\n  " + "\n  ".join(offences)
    )


def test_allowlist_entries_still_exist_and_are_still_needed():
    """The allowlist does not rot: each entry must exist and still omit an encoding."""
    for rel in ALLOWED:
        path = SRC_ROOT / rel
        assert path.is_file(), f"allowlist names a file that no longer exists: {rel}"

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        calls = _text_calls_without_encoding(tree)
        assert calls, (
            f"{rel} is allowlisted but no longer omits an encoding — drop the entry "
            "so the file is covered again"
        )
