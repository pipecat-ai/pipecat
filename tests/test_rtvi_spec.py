#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
import warnings
from typing import Any

from pydantic import TypeAdapter

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.processors.frameworks.rtvi.observer import RTVIObserverParams
from pipecat.processors.frameworks.rtvi.spec import (
    BEHAVIORS,
    GROUPS,
    MESSAGES,
    build_spec,
    contract_sha256,
    released,
    render_spec,
    spec_path,
)


def _undescribed(schema: Any, path: str = "") -> list[str]:
    """The fields in a schema, nested ones included, that have no description."""
    found = []
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "properties":
                for field, field_schema in value.items():
                    if "description" not in field_schema:
                        found.append(path + field)
                    found += _undescribed(field_schema, f"{path}{field}.")
            else:
                found += _undescribed(value, path)
    elif isinstance(schema, list):
        for value in schema:
            found += _undescribed(value, path)
    return found


class TestRTVISpec(unittest.TestCase):
    def test_shipped_spec_file_is_current(self):
        # The repository keeps the document for client SDKs to vendor. When the
        # regenerated document changes the wire format of a released version,
        # test_released_wire_formats_are_unchanged fails too.
        self.assertTrue(
            spec_path().read_text(encoding="utf-8") == render_spec(),
            f"spec/{RTVI.PROTOCOL_VERSION}/rtvi.asyncapi.json is out of date; run "
            "`uv run python scripts/rtvi/write_rtvi_spec.py`",
        )

    def test_released_wire_formats_are_unchanged(self):
        # A client implements a released version from its document, so its wire
        # format never changes; a change to the wire format is a new version.
        # Descriptions and examples can still be improved.
        for version, entry in released().items():
            with self.subTest(version):
                self.assertTrue(
                    spec_path(version).exists(),
                    f"spec/{version} was released in Pipecat {entry['pipecat']}",
                )
                self.assertEqual(
                    contract_sha256(version),
                    entry["contract_sha256"],
                    f"the contract of protocol {version} changed since Pipecat "
                    f"{entry['pipecat']} released it. For a change to the wire format, bump "
                    "PROTOCOL_VERSION in models.py and restore the released version. For a "
                    "schema that describes what the server already sends more precisely, "
                    "update contract_sha256 in spec/released.json (see spec/README.md)",
                )

    def test_message_types_are_unique(self):
        types = [spec.type for spec in MESSAGES]
        self.assertEqual(len(types), len(set(types)))

    def test_examples_match_their_data(self):
        for spec in MESSAGES:
            with self.subTest(spec.type):
                if spec.data is None:
                    self.assertIsNone(spec.example)
                else:
                    self.assertIsNotNone(spec.example)
                    # Deprecated messages' data models warn when built.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        TypeAdapter(spec.data).validate_python(spec.example)

    def test_enabled_by_names_an_observer_param(self):
        for spec in MESSAGES:
            if spec.enabled_by:
                with self.subTest(spec.type):
                    self.assertIn(spec.enabled_by, RTVIObserverParams.__dataclass_fields__)

    def test_document_version_is_the_protocol_version(self):
        self.assertEqual(build_spec()["info"]["version"], RTVI.PROTOCOL_VERSION)

    def test_operations_split_messages_by_direction(self):
        operations = build_spec()["operations"]
        received = len(operations["receiveClientMessages"]["messages"])
        sent = len(operations["sendServerMessages"]["messages"])
        self.assertEqual(received, sum(s.direction == "client-to-server" for s in MESSAGES))
        self.assertEqual(sent, sum(s.direction == "server-to-client" for s in MESSAGES))

    def test_behaviors_are_unique_and_name_known_messages(self):
        ids = [behavior.id for behavior in BEHAVIORS]
        self.assertEqual(len(ids), len(set(ids)))
        types = {spec.type for spec in MESSAGES}
        for behavior in BEHAVIORS:
            with self.subTest(behavior.id):
                self.assertEqual(set(behavior.messages) - types, set())

    def test_every_message_is_in_one_group(self):
        grouped = [type for group in GROUPS for type in group.messages]
        self.assertEqual(sorted(grouped), sorted(spec.type for spec in MESSAGES))

    def test_every_field_has_a_description(self):
        # A client's author reads each field's description, in the document and
        # on the docs site. Describe a field in the Parameters: section of its
        # model's docstring, and a message's free-form data in its message model's.
        document = build_spec()
        undescribed = []
        for name, schema in document["components"]["schemas"].items():
            undescribed += [f"{name}.{field}" for field in _undescribed(schema)]
        for name, message in document["components"]["messages"].items():
            data = message["payload"]["properties"].get("data")
            if data is None:
                continue
            if not {"$ref", "anyOf", "description"} & data.keys():
                undescribed.append(f"{name}.data")
            undescribed += [f"{name}.data.{field}" for field in _undescribed(data)]
        self.assertEqual(undescribed, [])

    def test_responses_carry_an_id(self):
        specs = {spec.type: spec for spec in MESSAGES}
        for type in ("bot-ready", "server-response", "error-response"):
            with self.subTest(type):
                self.assertTrue(specs[type].has_id)


if __name__ == "__main__":
    unittest.main()
