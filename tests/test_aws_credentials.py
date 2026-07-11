#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for AWS shared credential resolution."""

import os
import unittest
from unittest.mock import MagicMock, patch

from pipecat.services.aws.utils import AWSCredentials, resolve_credentials


class TestResolveCredentials(unittest.TestCase):
    """Tests for resolve_credentials() fallback chain."""

    def test_explicit_credentials_take_priority(self):
        """Explicit parameters override env vars and botocore chain."""
        result = resolve_credentials(
            aws_access_key_id="explicit-key",
            aws_secret_access_key="explicit-secret",
            aws_session_token="explicit-token",
            region="eu-west-1",
        )
        self.assertEqual(result.access_key, "explicit-key")
        self.assertEqual(result.secret_key, "explicit-secret")
        self.assertEqual(result.session_token, "explicit-token")
        self.assertEqual(result.region, "eu-west-1")

    @patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "env-key",
            "AWS_SECRET_ACCESS_KEY": "env-secret",
            "AWS_SESSION_TOKEN": "env-token",
            "AWS_REGION": "ap-southeast-2",
        },
    )
    def test_env_vars_fallback(self):
        """Environment variables are used when explicit params are None."""
        result = resolve_credentials()
        self.assertEqual(result.access_key, "env-key")
        self.assertEqual(result.secret_key, "env-secret")
        self.assertEqual(result.session_token, "env-token")
        self.assertEqual(result.region, "ap-southeast-2")

    @patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "env-key",
            "AWS_SECRET_ACCESS_KEY": "env-secret",
        },
    )
    def test_explicit_overrides_env(self):
        """Explicit params win over environment variables."""
        result = resolve_credentials(
            aws_access_key_id="override-key",
            aws_secret_access_key="override-secret",
        )
        self.assertEqual(result.access_key, "override-key")
        self.assertEqual(result.secret_key, "override-secret")

    @patch.dict(os.environ, {}, clear=True)
    def test_partial_explicit_credentials_do_not_mix_with_botocore_chain(self):
        """Partial explicit credentials are not completed from ambient botocore credentials."""
        mock_frozen = MagicMock()
        mock_frozen.access_key = "ambient-key"
        mock_frozen.secret_key = "ambient-secret"
        mock_frozen.token = "ambient-token"

        mock_creds = MagicMock()
        mock_creds.get_frozen_credentials.return_value = mock_frozen

        mock_session = MagicMock()
        mock_session.get_credentials.return_value = mock_creds

        with patch("botocore.session.Session", return_value=mock_session) as mock_session_cls:
            result = resolve_credentials(aws_access_key_id="explicit-key")

        self.assertEqual(result.access_key, "explicit-key")
        self.assertIsNone(result.secret_key)
        mock_session_cls.assert_not_called()

    @patch.dict(os.environ, {}, clear=True)
    def test_botocore_chain_fallback(self):
        """When no explicit creds or env vars, falls back to botocore chain."""
        mock_frozen = MagicMock()
        mock_frozen.access_key = "ambient-key"
        mock_frozen.secret_key = "ambient-secret"
        mock_frozen.token = "ambient-token"

        mock_creds = MagicMock()
        mock_creds.get_frozen_credentials.return_value = mock_frozen

        mock_session = MagicMock()
        mock_session.get_credentials.return_value = mock_creds

        with patch("botocore.session.Session", return_value=mock_session):
            result = resolve_credentials()

        self.assertEqual(result.access_key, "ambient-key")
        self.assertEqual(result.secret_key, "ambient-secret")
        self.assertEqual(result.session_token, "ambient-token")
        mock_session.set_config_variable.assert_called_once_with("region", "us-east-1")

    @patch.dict(os.environ, {}, clear=True)
    def test_default_region(self):
        """Default region is us-east-1 when nothing is specified."""
        result = resolve_credentials(
            aws_access_key_id="key",
            aws_secret_access_key="secret",
        )
        self.assertEqual(result.region, "us-east-1")

    def test_returns_aws_credentials_dataclass(self):
        """Result is an AWSCredentials instance."""
        result = resolve_credentials(
            aws_access_key_id="key",
            aws_secret_access_key="secret",
        )
        self.assertIsInstance(result, AWSCredentials)

    @patch.dict(os.environ, {}, clear=True)
    def test_none_when_no_credentials_available(self):
        """access_key and secret_key are None when nothing resolves."""
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = None

        with patch("botocore.session.Session", return_value=mock_session):
            result = resolve_credentials()

        self.assertIsInstance(result, AWSCredentials)
        self.assertIsNone(result.access_key)
        self.assertIsNone(result.secret_key)

    @patch.dict(os.environ, {}, clear=True)
    def test_botocore_import_error_returns_none_credentials(self):
        """ImportError from botocore is swallowed; result has None credentials."""
        with patch("botocore.session.Session", side_effect=ImportError("no botocore")):
            result = resolve_credentials()

        self.assertIsInstance(result, AWSCredentials)
        self.assertIsNone(result.access_key)
        self.assertIsNone(result.secret_key)


if __name__ == "__main__":
    unittest.main()
