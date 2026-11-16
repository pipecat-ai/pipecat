#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for AWS credential resolution and renewal."""

import datetime
import os
import unittest
from unittest.mock import MagicMock, patch

import aiobotocore.session
import pytest
from aiobotocore.credentials import AioRefreshableCredentials

from pipecat.services.aws.agent_core import AWSAgentCoreProcessor
from pipecat.services.aws.llm import AWSBedrockLLMService
from pipecat.services.aws.stt import AWSTranscribeSTTService
from pipecat.services.aws.tts import AWSPollyTTSService
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
    def test_no_keys_when_nothing_configured(self):
        """With nothing configured, no keys are returned and the default chain isn't read."""
        with patch("botocore.session.Session") as mock_session_cls:
            result = resolve_credentials()

        self.assertIsNone(result.access_key)
        self.assertIsNone(result.secret_key)
        self.assertIsNone(result.session_token)
        mock_session_cls.assert_not_called()

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


def _renewing_credentials() -> AioRefreshableCredentials:
    """Temporary credentials that are always near expiry, so every use renews them."""
    renewals = 0

    async def renew():
        nonlocal renewals
        renewals += 1
        expiry = datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=1)
        return {
            "access_key": f"key-{renewals}",
            "secret_key": f"secret-{renewals}",
            "token": f"token-{renewals}",
            "expiry_time": expiry.isoformat(),
        }

    return AioRefreshableCredentials(
        access_key="key-0",
        secret_key="secret-0",
        token="token-0",
        expiry_time=datetime.datetime.now(datetime.UTC),
        refresh_using=renew,
        method="test",
    )


@pytest.fixture
def no_configured_keys(monkeypatch):
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_service,client_name",
    [
        (lambda: AWSBedrockLLMService(), "bedrock-runtime"),
        (lambda: AWSPollyTTSService(), "polly"),
        (
            lambda: AWSAgentCoreProcessor(
                agentArn="arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/test"
            ),
            "bedrock-agentcore",
        ),
    ],
    ids=["bedrock", "polly", "agentcore"],
)
async def test_clients_sign_with_renewed_default_chain_credentials(
    no_configured_keys, make_service, client_name
):
    """Without configured keys, each client signs with the session's current credentials."""
    # The default chain has credentials while the service is built, so a copy
    # taken then would be what every client signs with.
    snapshot = MagicMock(access_key="snapshot", secret_key="snapshot", token="snapshot")
    with patch("botocore.session.Session") as session_cls:
        session_cls.return_value.get_credentials.return_value.get_frozen_credentials.return_value = snapshot
        service = make_service()
    service._aws_session._credentials = _renewing_credentials()

    signing_keys = []
    for _ in range(2):
        async with service._aws_session.create_client(client_name, **service._aws_params) as client:
            frozen = await client._request_signer._credentials.get_frozen_credentials()
            signing_keys.append(frozen.access_key)

    assert signing_keys[0] != signing_keys[1]
    assert all(key.startswith("key-") for key in signing_keys)


@pytest.mark.asyncio
async def test_transcribe_signs_each_connection_with_current_keys(no_configured_keys):
    """Without configured keys, Transcribe signs every connection with renewed keys."""
    service = AWSTranscribeSTTService()
    service._aws_session = aiobotocore.session.get_session()
    service._aws_session._credentials = _renewing_credentials()

    first = await service._signing_credentials()
    second = await service._signing_credentials()

    assert first["access_key"] != second["access_key"]
    assert second["session_token"] == second["access_key"].replace("key", "token")


@pytest.mark.asyncio
async def test_transcribe_signs_with_configured_keys_as_given(no_configured_keys):
    """Configured keys are used without consulting the default chain."""
    service = AWSTranscribeSTTService(
        api_key="secret", aws_access_key_id="key", aws_session_token="token"
    )

    credentials = await service._signing_credentials()

    assert credentials == {"access_key": "key", "secret_key": "secret", "session_token": "token"}
    assert service._aws_session is None


@pytest.mark.asyncio
async def test_transcribe_without_any_credentials_raises(no_configured_keys):
    """With no keys anywhere, signing fails with a clear error."""
    service = AWSTranscribeSTTService()
    service._aws_session = aiobotocore.session.get_session()

    async def no_credentials():
        return None

    service._aws_session.get_credentials = no_credentials

    with pytest.raises(ValueError, match="No AWS credentials found"):
        await service._signing_credentials()


if __name__ == "__main__":
    unittest.main()
