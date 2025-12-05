import importlib.util
import sys
import unittest
from unittest.mock import MagicMock

# Mock Pipecat package
sys.modules["pipecat"] = MagicMock()
sys.modules["pipecat"].__version__ = "0.0.0-test"

# Load the module directly from source
spec = importlib.util.spec_from_file_location(
    "pipecat.services.google.utils", "src/pipecat/services/google/utils.py"
)
utils_module = importlib.util.module_from_spec(spec)
sys.modules["pipecat.services.google.utils"] = utils_module
spec.loader.exec_module(utils_module)

update_google_client_http_options = utils_module.update_google_client_http_options
pipecat_version = "0.0.0-test"


class TestGoogleUtils(unittest.TestCase):
    def test_update_google_client_http_options_none(self):
        options = update_google_client_http_options(None)
        self.assertEqual(options, {"headers": {"x-goog-api-client": f"pipecat/{pipecat_version}"}})

    def test_update_google_client_http_options_dict_empty(self):
        options = update_google_client_http_options({})
        self.assertEqual(options, {"headers": {"x-goog-api-client": f"pipecat/{pipecat_version}"}})

    def test_update_google_client_http_options_dict_existing_headers(self):
        initial_options = {"headers": {"Authorization": "Bearer token"}}
        options = update_google_client_http_options(initial_options)
        self.assertEqual(options["headers"]["Authorization"], "Bearer token")
        self.assertEqual(options["headers"]["x-goog-api-client"], f"pipecat/{pipecat_version}")

    def test_update_google_client_http_options_object(self):
        class HttpOptions:
            def __init__(self):
                self.headers = None

        http_options = HttpOptions()
        updated_options = update_google_client_http_options(http_options)
        self.assertEqual(
            updated_options.headers, {"x-goog-api-client": f"pipecat/{pipecat_version}"}
        )

    def test_update_google_client_http_options_object_existing_headers(self):
        class HttpOptions:
            def __init__(self):
                self.headers = {"Authorization": "Bearer token"}

        http_options = HttpOptions()
        updated_options = update_google_client_http_options(http_options)
        self.assertEqual(updated_options.headers["Authorization"], "Bearer token")
        self.assertEqual(updated_options.headers["x-goog-api-client"], f"pipecat/{pipecat_version}")


if __name__ == "__main__":
    unittest.main()
