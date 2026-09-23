#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.runner.run import _run_sip
from pipecat.runner.sip import (
    SIPClientConfig,
    cleanup,
    configure,
    resolve_media_nat_params,
)
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


class TestResolveMediaNatParams(unittest.TestCase):
    STUN = ("medianat=stun", "stunserver=stun:stun.l.google.com:19302")

    def test_explicit_params_win_verbatim(self):
        # SIP_EXTRA_PARAMS takes over unchanged; no default is added.
        result = resolve_media_nat_params("medianat=ice,stunserver=stun:x:3478")
        self.assertEqual(result, ("medianat=ice", "stunserver=stun:x:3478"))

    def test_default_enables_stun(self):
        with patch.dict("os.environ", {}, clear=True):
            result = resolve_media_nat_params(None)

        self.assertEqual(result, self.STUN)

    def test_stun_server_off_disables(self):
        with patch.dict("os.environ", {"SIP_STUN_SERVER": "off"}, clear=True):
            result = resolve_media_nat_params(None)

        self.assertIsNone(result)

    def test_custom_stun_server(self):
        with patch.dict(
            "os.environ", {"SIP_STUN_SERVER": "stun:stun.example.com:3478"}, clear=True
        ):
            result = resolve_media_nat_params(None)

        self.assertEqual(result, ("medianat=stun", "stunserver=stun:stun.example.com:3478"))

    def test_bare_stun_server_gets_scheme(self):
        # A server given without a URI scheme is normalized to stun:host:port.
        with patch.dict("os.environ", {"SIP_STUN_SERVER": "stun.example.com:3478"}, clear=True):
            result = resolve_media_nat_params(None)

        self.assertEqual(result, ("medianat=stun", "stunserver=stun:stun.example.com:3478"))


PROVISIONED_CONFIG = SIPClientConfig(
    user="pipecat-abcd1234",
    password="s3cretpass12",
    domain="mydomain.sip-us.daily.co",
    transport="tls",
    sip_uri="sip:pipecat-abcd1234@mydomain.sip-us.daily.co",
    provisioned=True,
)


class TestRunSip(unittest.IsolatedAsyncioTestCase):
    """_run_sip: env parsing and the cleanup-in-finally contract."""

    def _session_cm(self):
        # aiohttp.ClientSession() used as an async context manager.
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=AsyncMock())
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    def _patches(self, env, bot):
        return (
            patch.dict("os.environ", env, clear=True),
            patch("pipecat.runner.run.aiohttp.ClientSession", return_value=self._session_cm()),
            patch("pipecat.runner.sip.configure", AsyncMock(return_value=PROVISIONED_CONFIG)),
            patch("pipecat.runner.sip.cleanup", AsyncMock()),
            patch("pipecat.runner.run._get_bot_module", return_value=bot),
        )

    async def test_non_integer_reg_interval_exits(self):
        bot = MagicMock()
        bot.bot = AsyncMock()
        env_patch, session_patch, configure_patch, cleanup_patch, bot_patch = self._patches(
            {"SIP_REG_INTERVAL": "not-an-int"}, bot
        )
        with env_patch, session_patch, configure_patch, cleanup_patch as cleanup_mock, bot_patch:
            with self.assertRaises(SystemExit):
                await _run_sip(argparse.Namespace(runner_body=None))

        bot.bot.assert_not_awaited()
        cleanup_mock.assert_not_awaited()  # failed before provisioning's try/finally

    async def test_cleanup_runs_when_bot_raises(self):
        bot = MagicMock()
        bot.bot = AsyncMock(side_effect=RuntimeError("bot boom"))
        env_patch, session_patch, configure_patch, cleanup_patch, bot_patch = self._patches(
            {"SIP_REG_INTERVAL": "900", "SIP_RTP_TIMEOUT": "45"}, bot
        )
        with env_patch, session_patch, configure_patch, cleanup_patch as cleanup_mock, bot_patch:
            with self.assertRaises(RuntimeError):
                await _run_sip(argparse.Namespace(runner_body=None))

        cleanup_mock.assert_awaited_once()  # cleanup runs in finally even on bot error
        runner_args = bot.bot.await_args.args[0]
        self.assertEqual(runner_args.reg_interval, 900)
        self.assertEqual(runner_args.rtp_timeout, 45)
