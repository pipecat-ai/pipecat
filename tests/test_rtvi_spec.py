#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
import warnings

from pydantic import TypeAdapter

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.processors.frameworks.rtvi.observer import RTVIObserverParams
from pipecat.processors.frameworks.rtvi.spec import MESSAGES, build_spec, render_spec, spec_path


class TestRTVISpec(unittest.TestCase):
    def test_shipped_spec_file_is_current(self):
        # The repository keeps the document for client SDKs to vendor.
        # Regenerate it with `uv run python scripts/rtvi/write_rtvi_spec.py`.
        self.assertEqual(
            spec_path().read_text(encoding="utf-8"),
            render_spec(),
            f"spec/{RTVI.PROTOCOL_VERSION}/rtvi.asyncapi.json is out of date; run "
            "`uv run python scripts/rtvi/write_rtvi_spec.py`",
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

    def test_responses_carry_an_id(self):
        specs = {spec.type: spec for spec in MESSAGES}
        for type in ("bot-ready", "server-response", "error-response"):
            with self.subTest(type):
                self.assertTrue(specs[type].has_id)


if __name__ == "__main__":
    unittest.main()
