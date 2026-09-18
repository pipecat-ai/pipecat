#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from unittest.mock import AsyncMock, patch

from pipecat.runner.sip import SIPClientConfig, cleanup, configure
from pipecat.transports.daily.utils import DailySIPClientObject

CLIENT = DailySIPClientObject(
    username="pipecat-abcd1234",
    domain="mydomain.sip-us.daily.co",
    sip_uri="sip:pipecat-abcd1234@mydomain.sip-us.daily.co",
    password="s3cretpass12",
    expires_at="2026-09-18T12:00:00.000Z",
)

ENV_ACCOUNT = {
    "SIP_USER": "1001",
    "SIP_PASS": "secret",
    "SIP_DOMAIN": "sip.example.com",
}


class TestConfigure(unittest.IsolatedAsyncioTestCase):
    async def test_env_account_used_as_is(self):
        with (
            patch.dict("os.environ", ENV_ACCOUNT, clear=True),
            patch("pipecat.runner.sip.DailyRESTHelper") as helper_cls,
        ):
            config = await configure(AsyncMock())

        helper_cls.assert_not_called()
        self.assertEqual(config.user, "1001")
        self.assertEqual(config.password, "secret")
        self.assertEqual(config.domain, "sip.example.com")
        self.assertEqual(config.transport, "udp")
        self.assertIsNone(config.sip_uri)
        self.assertFalse(config.provisioned)

    async def test_provisions_ephemeral_client(self):
        with (
            patch.dict("os.environ", {"DAILY_API_KEY": "test-key"}, clear=True),
            patch("pipecat.runner.sip.DailyRESTHelper") as helper_cls,
        ):
            helper = helper_cls.return_value
            helper.create_sip_client = AsyncMock(return_value=CLIENT)

            config = await configure(AsyncMock(), client_exp_duration=1.0)

        params = helper.create_sip_client.call_args.args[0]
        self.assertTrue(params.username.startswith("pipecat-"))
        self.assertEqual(params.expires_in_seconds, 3600)
        self.assertEqual(config.user, "pipecat-abcd1234")
        self.assertEqual(config.password, "s3cretpass12")
        self.assertEqual(config.domain, "mydomain.sip-us.daily.co")
        self.assertEqual(config.transport, "tls")
        self.assertEqual(config.sip_uri, "sip:pipecat-abcd1234@mydomain.sip-us.daily.co")
        self.assertTrue(config.provisioned)

    async def test_provisions_with_custom_prefix(self):
        with (
            patch.dict("os.environ", {"DAILY_API_KEY": "test-key"}, clear=True),
            patch("pipecat.runner.sip.DailyRESTHelper") as helper_cls,
        ):
            helper = helper_cls.return_value
            helper.create_sip_client = AsyncMock(return_value=CLIENT)

            await configure(AsyncMock(), prefix="mybot")

        params = helper.create_sip_client.call_args.args[0]
        self.assertTrue(params.username.startswith("mybot-"))

    async def test_no_account_and_no_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(Exception) as ctx:
                await configure(AsyncMock())

        self.assertIn("DAILY_API_KEY", str(ctx.exception))


class TestCleanup(unittest.IsolatedAsyncioTestCase):
    async def test_deletes_provisioned_client(self):
        config = SIPClientConfig(
            user="pipecat-abcd1234",
            password="s3cretpass12",
            domain="mydomain.sip-us.daily.co",
            transport="tls",
            provisioned=True,
        )
        with (
            patch.dict("os.environ", {"DAILY_API_KEY": "test-key"}, clear=True),
            patch("pipecat.runner.sip.DailyRESTHelper") as helper_cls,
        ):
            helper = helper_cls.return_value
            helper.delete_sip_client = AsyncMock(return_value=True)

            await cleanup(AsyncMock(), config)

        helper.delete_sip_client.assert_awaited_once_with("pipecat-abcd1234")

    async def test_env_account_is_left_alone(self):
        config = SIPClientConfig(
            user="1001", password="secret", domain="sip.example.com", transport="udp"
        )
        with (
            patch.dict("os.environ", {"DAILY_API_KEY": "test-key"}, clear=True),
            patch("pipecat.runner.sip.DailyRESTHelper") as helper_cls,
        ):
            await cleanup(AsyncMock(), config)

        helper_cls.assert_not_called()

    async def test_delete_failure_does_not_raise(self):
        config = SIPClientConfig(
            user="pipecat-abcd1234",
            password="s3cretpass12",
            domain="mydomain.sip-us.daily.co",
            transport="tls",
            provisioned=True,
        )
        with (
            patch.dict("os.environ", {"DAILY_API_KEY": "test-key"}, clear=True),
            patch("pipecat.runner.sip.DailyRESTHelper") as helper_cls,
        ):
            helper = helper_cls.return_value
            helper.delete_sip_client = AsyncMock(side_effect=Exception("boom"))

            await cleanup(AsyncMock(), config)  # must not raise
