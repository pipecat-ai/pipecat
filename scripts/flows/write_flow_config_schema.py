#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Write the JSON Schema for ``FlowConfig`` into the ``pipecat.flows`` package.

The schema ships with Pipecat as ``pipecat/flows/flow_config.schema.json`` so
editors and flow builders can validate and autocomplete flow config files.
``tests/test_flows_config.py`` fails when the file drifts from the model;
rerun this script to refresh it::

    uv run python scripts/flows/write_flow_config_schema.py
"""

import json
from pathlib import Path

from pipecat.flows import FlowConfig

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "src/pipecat/flows/flow_config.schema.json"

# Pydantic emits draft 2020-12 keywords; the declaration lets validators and
# editors apply that draft's rules without guessing.
JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def render_schema() -> str:
    """The schema file's content."""
    schema = {"$schema": JSON_SCHEMA_DIALECT, **FlowConfig.model_json_schema()}
    return json.dumps(schema, indent=2) + "\n"


if __name__ == "__main__":
    SCHEMA_PATH.write_text(render_schema(), encoding="utf-8")
    print(f"wrote {SCHEMA_PATH}")
