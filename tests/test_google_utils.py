#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import unittest
from unittest.mock import patch

from google.auth.exceptions import DefaultCredentialsError

import pipecat.services.google.utils
from pipecat.services.google.utils import _vertex_credentials, update_google_client_http_options

MOCKED_VERSION = "0.0.0-test"

pipecat.services.google.utils.pipecat_version = lambda: MOCKED_VERSION


class TestGoogleUtils(unittest.TestCase):
    def test_update_google_client_http_options_none(self):
        options = update_google_client_http_options(None)
        self.assertEqual(options, {"headers": {"x-goog-api-client": f"pipecat/{MOCKED_VERSION}"}})

    def test_update_google_client_http_options_dict_empty(self):
        options = update_google_client_http_options({})
        self.assertEqual(options, {"headers": {"x-goog-api-client": f"pipecat/{MOCKED_VERSION}"}})

    def test_update_google_client_http_options_dict_existing_headers(self):
        initial_options = {"headers": {"Authorization": "Bearer token"}}
        options = update_google_client_http_options(initial_options)
        self.assertEqual(options["headers"]["Authorization"], "Bearer token")
        self.assertEqual(options["headers"]["x-goog-api-client"], f"pipecat/{MOCKED_VERSION}")

    def test_update_google_client_http_options_object(self):
        class HttpOptions:
            def __init__(self):
                self.headers = None

        http_options = HttpOptions()
        updated_options = update_google_client_http_options(http_options)
        self.assertEqual(
            updated_options.headers, {"x-goog-api-client": f"pipecat/{MOCKED_VERSION}"}
        )

    def test_update_google_client_http_options_object_existing_headers(self):
        class HttpOptions:
            def __init__(self):
                self.headers = {"Authorization": "Bearer token"}

        http_options = HttpOptions()
        updated_options = update_google_client_http_options(http_options)
        self.assertEqual(updated_options.headers["Authorization"], "Bearer token")
        self.assertEqual(updated_options.headers["x-goog-api-client"], f"pipecat/{MOCKED_VERSION}")


class TestVertexCredentials(unittest.TestCase):
    def test_service_account_json(self):
        """Test credentials loaded from service account JSON, scoped and refreshed."""
        info = {"type": "service_account", "project_id": "p"}
        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_info"
        ) as from_info:
            creds = _vertex_credentials(json.dumps(info), None)

        self.assertIs(creds, from_info.return_value)
        self.assertEqual(from_info.call_args.args[0], info)
        self.assertEqual(
            from_info.call_args.kwargs["scopes"],
            ["https://www.googleapis.com/auth/cloud-platform"],
        )
        creds.refresh.assert_called_once()

    def test_no_credentials(self):
        """Test a clear error when no credentials are given and none are found."""
        with patch("google.auth.default", side_effect=DefaultCredentialsError("none")):
            with self.assertRaisesRegex(ValueError, "No valid credentials provided"):
                _vertex_credentials(None, None)


if __name__ == "__main__":
    unittest.main()
