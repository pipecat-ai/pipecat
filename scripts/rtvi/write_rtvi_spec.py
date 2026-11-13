#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Write the current RTVI protocol version's AsyncAPI document.

The document is kept in the repository as
``src/pipecat/processors/frameworks/rtvi/spec/<version>/rtvi.asyncapi.json``
so client SDKs can generate their protocol code from it at a pinned version.
Documents of earlier versions stay as they were. ``tests/test_rtvi_spec.py``
fails when the current version's file drifts from the models; rerun this
script to refresh it::

    uv run python scripts/rtvi/write_rtvi_spec.py
"""

from pipecat.processors.frameworks.rtvi.spec import render_spec, spec_path

if __name__ == "__main__":
    path = spec_path()
    path.parent.mkdir(exist_ok=True)
    path.write_text(render_spec(), encoding="utf-8")
    print(f"wrote {path}")
