#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Record the current RTVI protocol version as released.

Adds the current protocol version to
``src/pipecat/processors/frameworks/rtvi/spec/released.json``, with a hash of
its wire format and the Pipecat release that ships it, unless an earlier
release already recorded it. ``tests/test_rtvi_spec.py`` then fails when the
wire format of a recorded version changes. The changelog workflow runs it at
release-prep, with the release being prepared::

    uv run python scripts/rtvi/record_released_spec.py --version 1.17.0
"""

import argparse
import json
import sys

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.processors.frameworks.rtvi.spec import (
    contract_sha256,
    released,
    released_path,
    render_spec,
    spec_path,
)


def _version_key(version: str) -> list[int]:
    return [int(part) for part in version.split(".")]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record the current RTVI protocol version as released."
    )
    parser.add_argument("--version", required=True, help="the Pipecat release being prepared")
    args = parser.parse_args()

    version = RTVI.PROTOCOL_VERSION
    record = released()
    if version in record:
        print(f"RTVI {version} was released in Pipecat {record[version]['pipecat']}")
        return 0
    if not spec_path().exists() or spec_path().read_text(encoding="utf-8") != render_spec():
        print(
            f"spec/{version}/rtvi.asyncapi.json is out of date; run "
            "`uv run python scripts/rtvi/write_rtvi_spec.py`",
            file=sys.stderr,
        )
        return 1

    record[version] = {"contract_sha256": contract_sha256(), "pipecat": args.version}
    record = dict(sorted(record.items(), key=lambda item: _version_key(item[0])))
    released_path().write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"recorded RTVI {version} as released in Pipecat {args.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
