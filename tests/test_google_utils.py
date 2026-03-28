#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

import pipecat.services.google.utils
from pipecat.services.google.utils import update_google_client_http_options

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


if __name__ == "__main__":
    unittest.main()
