#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import json
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.runner.run import _run_sip
from pipecat.runner.sip import (
    SIPClientConfig,
    cleanup,
    configure,
    parse_jitter_buffer,
    resolve_media_nat_params,
    sip_runner_arguments,
)
from pipecat.runner.types import RunnerArguments, SIPRunnerArguments
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

# The sip_client block of a Daily SIP-trunk notification, as pluot-core sends it.
BODY_CLIENT = {
    "username": "trunk-0a1b2c3d-9f8e7d6c5b4a3f2e",
    "password": "per-call-secret",
    "domain": "acme.sip-us.daily.co",
    "sip_uri": "sip:trunk-0a1b2c3d-9f8e7d6c5b4a3f2e@acme.sip-us.daily.co",
    "expires_at": "2026-10-08T12:00:00.000Z",
    "transport": "udp",
    "preferred_registrar_ip": "10.0.0.5",
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

    async def test_body_account_wins_over_env_and_never_provisions(self):
        env = {**ENV_ACCOUNT, "DAILY_API_KEY": "test-key"}
        with (
            patch.dict("os.environ", env, clear=True),
            patch("pipecat.runner.sip.DailyRESTHelper") as helper_cls,
        ):
            config = await configure(AsyncMock(), body={"sip_client": BODY_CLIENT})

        helper_cls.assert_not_called()
        self.assertEqual(config.user, BODY_CLIENT["username"])
        self.assertEqual(config.password, "per-call-secret")
        self.assertEqual(config.domain, "acme.sip-us.daily.co")
        self.assertEqual(config.transport, "udp")
        self.assertEqual(config.sip_uri, BODY_CLIENT["sip_uri"])
        self.assertFalse(config.provisioned)

    async def test_body_password_optional_and_sip_uri_composed(self):
        client = {"username": "u1", "domain": "d.example"}
        with patch.dict("os.environ", {}, clear=True):
            config = await configure(AsyncMock(), body={"sip_client": client})

        self.assertEqual(config.password, "")
        self.assertEqual(config.sip_uri, "sip:u1@d.example")

    async def test_body_transport_precedence(self):
        client = {"username": "u1", "domain": "d.example"}
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                (await configure(AsyncMock(), body={"sip_client": client})).transport, "tls"
            )
        with patch.dict("os.environ", {"SIP_TRANSPORT": "tcp"}, clear=True):
            self.assertEqual(
                (await configure(AsyncMock(), body={"sip_client": client})).transport, "tcp"
            )
            self.assertEqual(
                (
                    await configure(
                        AsyncMock(), body={"sip_client": {**client, "transport": "udp"}}
                    )
                ).transport,
                "udp",
            )

    async def test_body_without_a_usable_account_falls_through(self):
        for body in (None, {}, {"sip_client": None}, {"sip_client": {"username": "only"}}):
            with (
                self.subTest(body=body),
                patch.dict("os.environ", ENV_ACCOUNT, clear=True),
                patch("pipecat.runner.sip.DailyRESTHelper") as helper_cls,
            ):
                config = await configure(AsyncMock(), body=body)
                self.assertEqual(config.user, "1001")
                helper_cls.assert_not_called()


class TestSipRunnerArguments(unittest.TestCase):
    CONFIG = SIPClientConfig(
        user="u1", password="p1", domain="d.example", transport="tcp", sip_uri="sip:u1@d.example"
    )

    def test_maps_account_and_defaults_with_empty_environment(self):
        with patch.dict("os.environ", {}, clear=True):
            args = sip_runner_arguments(self.CONFIG)

        self.assertIsInstance(args, SIPRunnerArguments)
        self.assertEqual(
            (args.user, args.password, args.domain, args.transport),
            ("u1", "p1", "d.example", "tcp"),
        )
        self.assertIsNone(args.audio_codecs)
        self.assertIsNone(args.auth_user)
        self.assertEqual(
            args.extra_params, ("medianat=stun", "stunserver=stun:stun.l.google.com:19302")
        )
        self.assertEqual(args.reg_interval, 600)
        self.assertEqual(args.rtp_timeout, 0)
        self.assertIsNone(args.instance_id)
        self.assertEqual(args.native_log_level, "warning")
        self.assertFalse(args.sip_trace)
        self.assertIsNone(args.net_interface)
        self.assertIsNone(args.jitter_buffer_mode)
        self.assertIsNone(args.jitter_buffer_ms)
        self.assertEqual(args.body, {})
        self.assertIsNone(args.session_id)
        self.assertFalse(args.handle_sigint)
        self.assertIsNone(args.cli_args)

    def test_applies_every_environment_setting(self):
        env = {
            "SIP_AUDIO_CODECS": "opus/48000/2, PCMU/8000/1",
            "SIP_AUTH_USER": "digest-user",
            "SIP_EXTRA_PARAMS": "medianat=turn,stunserver=turn:t:3478",
            "SIP_REG_INTERVAL": "240",
            "SIP_RTP_TIMEOUT": "30",
            "SIP_INSTANCE_ID": "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0",
            "SIP_NATIVE_LOG_LEVEL": "debug",
            "SIP_TRACE": "true",
            "SIP_NET_INTERFACE": "127.0.0.1",
            "SIP_JITTER_BUFFER": "adaptive:20-100",
        }
        with patch.dict("os.environ", env, clear=True):
            args = sip_runner_arguments(self.CONFIG)

        self.assertEqual(args.audio_codecs, ("opus/48000/2", "PCMU/8000/1"))
        self.assertEqual(args.auth_user, "digest-user")
        self.assertEqual(args.extra_params, ("medianat=turn", "stunserver=turn:t:3478"))
        self.assertEqual(args.reg_interval, 240)
        self.assertEqual(args.rtp_timeout, 30)
        self.assertEqual(args.instance_id, "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0")
        self.assertEqual(args.native_log_level, "debug")
        self.assertTrue(args.sip_trace)
        self.assertEqual(args.net_interface, "127.0.0.1")
        self.assertEqual(args.jitter_buffer_mode, "adaptive")
        self.assertEqual(args.jitter_buffer_ms, (20, 100))

    def test_carries_the_session_over(self):
        base = RunnerArguments(
            body={"sip_client": BODY_CLIENT, "call": {"From": "+1"}}, session_id="call-1"
        )
        base.handle_sigint = True
        base.handle_sigterm = True
        base.pipeline_idle_timeout_secs = None
        base.cli_args = argparse.Namespace(verbose=1)
        with patch.dict("os.environ", {}, clear=True):
            args = sip_runner_arguments(self.CONFIG, base)

        self.assertIs(args.body, base.body)
        self.assertEqual(args.session_id, "call-1")
        self.assertTrue(args.handle_sigint)
        self.assertTrue(args.handle_sigterm)
        self.assertIsNone(args.pipeline_idle_timeout_secs)
        self.assertIs(args.cli_args, base.cli_args)

    def test_rejects_malformed_environment(self):
        with patch.dict("os.environ", {"SIP_REG_INTERVAL": "soon"}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                sip_runner_arguments(self.CONFIG)
        self.assertIn("SIP_REG_INTERVAL", str(ctx.exception))
        with patch.dict("os.environ", {"SIP_JITTER_BUFFER": "fixed:60-40"}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                sip_runner_arguments(self.CONFIG)
        self.assertIn("SIP_JITTER_BUFFER", str(ctx.exception))


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


class TestParseJitterBuffer(unittest.TestCase):
    def test_unset_or_empty_keeps_the_stack_default(self):
        self.assertEqual(parse_jitter_buffer(None), (None, None))
        self.assertEqual(parse_jitter_buffer("  "), (None, None))

    def test_off(self):
        self.assertEqual(parse_jitter_buffer("off"), ("off", None))

    def test_mode_with_range(self):
        self.assertEqual(parse_jitter_buffer("fixed:40-60"), ("fixed", (40, 60)))
        self.assertEqual(parse_jitter_buffer("Adaptive: 20 - 100"), ("adaptive", (20, 100)))

    def test_bare_range_means_fixed(self):
        self.assertEqual(parse_jitter_buffer("40-60"), ("fixed", (40, 60)))

    def test_rejects_malformed_values(self):
        for spec in ("sometimes:20-40", "fixed", "fixed:20", "fixed:60-40", "fixed:0-40", "20"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                parse_jitter_buffer(spec)


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

    async def test_malformed_jitter_buffer_exits(self):
        bot = MagicMock()
        bot.bot = AsyncMock()
        env_patch, session_patch, configure_patch, cleanup_patch, bot_patch = self._patches(
            {"SIP_JITTER_BUFFER": "fixed:60-40"}, bot
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
            {
                "SIP_REG_INTERVAL": "900",
                "SIP_RTP_TIMEOUT": "45",
                "SIP_NET_INTERFACE": "127.0.0.1",
                "SIP_JITTER_BUFFER": "adaptive:20-100",
            },
            bot,
        )
        with env_patch, session_patch, configure_patch, cleanup_patch as cleanup_mock, bot_patch:
            with self.assertRaises(RuntimeError):
                await _run_sip(argparse.Namespace(runner_body=None))

        cleanup_mock.assert_awaited_once()  # cleanup runs in finally even on bot error
        runner_args = bot.bot.await_args.args[0]
        self.assertEqual(runner_args.reg_interval, 900)
        self.assertEqual(runner_args.rtp_timeout, 45)
        self.assertEqual(runner_args.net_interface, "127.0.0.1")
        self.assertEqual(runner_args.jitter_buffer_mode, "adaptive")
        self.assertEqual(runner_args.jitter_buffer_ms, (20, 100))
        self.assertTrue(runner_args.handle_sigint)

    async def test_runner_body_file_reaches_configure_and_the_bot(self):
        bot = MagicMock()
        bot.bot = AsyncMock()
        body = {"sip_client": BODY_CLIENT, "dialout": {"to": "sip:callee@example.com"}}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(body, f)
        env_patch, session_patch, configure_patch, cleanup_patch, bot_patch = self._patches({}, bot)
        with env_patch, session_patch, configure_patch as configure_mock, cleanup_patch, bot_patch:
            await _run_sip(argparse.Namespace(runner_body=f.name))

        self.assertEqual(configure_mock.await_args.kwargs["body"], body)
        runner_args = bot.bot.await_args.args[0]
        self.assertEqual(runner_args.body, body)
        self.assertIsNotNone(runner_args.session_id)
