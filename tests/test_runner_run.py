#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
from starlette.testclient import WebSocketDisconnect

from pipecat.runner.run import (
    _extract_ws_token,
    _generate_ws_token,
    _parse_ice_servers,
    _print_startup_message,
    _setup_daily_routes,
    _setup_telephony_routes,
    _setup_unified_start_route,
    _setup_webrtc_routes,
    _setup_websocket_routes,
    _transport_route_dependencies,
    _transport_routes_enabled,
    _verify_and_consume_ws_token,
)

try:
    import jwt as _jwt  # noqa: F401
    from livekit import api as _livekit_api  # noqa: F401

    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False


def _fake_smallwebrtc_modules(handler_kwargs: list | None = None) -> dict:
    """Build stand-in smallwebrtc modules so WebRTC routes can be set up without aiortc.

    Args:
        handler_kwargs: Optional list that collects the keyword arguments each
            ``SmallWebRTCRequestHandler`` is constructed with.

    Returns:
        Module map suitable for ``patch.dict(sys.modules, ...)``.
    """
    connection_module = types.ModuleType("pipecat.transports.smallwebrtc.connection")
    connection_module.SmallWebRTCConnection = MagicMock()

    @dataclass
    class IceServer:
        urls: str | list[str]
        username: str | None = None
        credential: str | None = None

    connection_module.IceServer = IceServer

    request_handler_module = types.ModuleType("pipecat.transports.smallwebrtc.request_handler")

    class IceCandidate(BaseModel):
        candidate: str
        sdp_mid: str
        sdp_mline_index: int

    class SmallWebRTCPatchRequest(BaseModel):
        pc_id: str
        candidates: list[IceCandidate] = []

    class SmallWebRTCRequest(BaseModel):
        sdp: str
        type: str
        pc_id: str | None = None
        restart_pc: bool | None = None
        request_data: dict | None = None

    class SmallWebRTCRequestHandler:
        def __init__(self, *args, **kwargs):
            if handler_kwargs is not None:
                handler_kwargs.append(kwargs)

        async def close(self):
            pass

    request_handler_module.IceCandidate = IceCandidate
    request_handler_module.SmallWebRTCPatchRequest = SmallWebRTCPatchRequest
    request_handler_module.SmallWebRTCRequest = SmallWebRTCRequest
    request_handler_module.SmallWebRTCRequestHandler = SmallWebRTCRequestHandler

    return {
        "pipecat.transports.smallwebrtc.connection": connection_module,
        "pipecat.transports.smallwebrtc.request_handler": request_handler_module,
    }


class TestRunnerRun(unittest.TestCase):
    def _capture_startup_message(self, args: argparse.Namespace) -> str:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _print_startup_message(args)
        return buffer.getvalue()

    def test_transport_route_dependencies_maps_transports_to_modules(self):
        self.assertEqual(_transport_route_dependencies("daily"), ("daily",))
        self.assertEqual(_transport_route_dependencies("webrtc"), ("aiortc",))
        self.assertEqual(_transport_route_dependencies("websocket"), ("fastapi", "websockets"))
        self.assertEqual(_transport_route_dependencies("telephony"), ("fastapi", "websockets"))
        self.assertEqual(_transport_route_dependencies("twilio"), ("fastapi", "websockets"))
        self.assertEqual(_transport_route_dependencies("telnyx"), ("fastapi", "websockets"))
        self.assertEqual(_transport_route_dependencies("plivo"), ("fastapi", "websockets"))
        self.assertEqual(_transport_route_dependencies("exotel"), ("fastapi", "websockets"))
        self.assertEqual(_transport_route_dependencies("vonage"), ())
        self.assertEqual(_transport_route_dependencies("livekit"), ("livekit.api",))

    def test_transport_routes_enabled_maps_transports_to_dependency_checks(self):
        def module_available(module: str) -> bool:
            return module in {"fastapi", "websockets"}

        with patch("pipecat.runner.run._is_module_available", side_effect=module_available):
            self.assertFalse(_transport_routes_enabled("daily"))
            self.assertFalse(_transport_routes_enabled("webrtc"))
            self.assertTrue(_transport_routes_enabled("websocket"))
            self.assertTrue(_transport_routes_enabled("telephony"))
            self.assertTrue(_transport_routes_enabled("twilio"))
            self.assertTrue(_transport_routes_enabled("vonage"))

    def test_setup_webrtc_routes_skips_when_aiortc_is_missing(self):
        """WebRTC routes should be optional when the webrtc extra is not installed."""
        app = FastAPI()
        args = argparse.Namespace(folder=None, esp32=False, host="localhost", ice_servers=[])

        with (
            patch("pipecat.runner.run._transport_routes_enabled", return_value=False),
            patch("pipecat.runner.run.logger") as logger,
        ):
            _setup_webrtc_routes(app, args, {})

        paths = {route.path for route in app.routes}
        self.assertNotIn("/api/offer", paths)
        logger.info.assert_not_called()

    def test_setup_webrtc_routes_registers_routes_when_webrtc_is_available(self):
        """WebRTC routes should be registered when dependencies are available."""
        app = FastAPI()
        args = argparse.Namespace(folder=None, esp32=False, host="localhost", ice_servers=[])

        with (
            patch("pipecat.runner.run._transport_routes_enabled", return_value=True),
            patch.dict(sys.modules, _fake_smallwebrtc_modules()),
        ):
            _setup_webrtc_routes(app, args, {})

        paths = {route.path for route in app.routes}
        self.assertIn("/api/offer", paths)
        self.assertIn("/files/{filename:path}", paths)

    def test_setup_webrtc_routes_gives_handler_no_ice_servers_when_unconfigured(self):
        """The bot peer keeps its previous behaviour when nothing is configured."""
        app = FastAPI()
        args = argparse.Namespace(folder=None, esp32=False, host="localhost", ice_servers=[])
        handler_kwargs = []

        with (
            patch("pipecat.runner.run._transport_routes_enabled", return_value=True),
            patch.dict(sys.modules, _fake_smallwebrtc_modules(handler_kwargs)),
        ):
            _setup_webrtc_routes(app, args, {})

        self.assertIsNone(handler_kwargs[0]["ice_servers"])

    def test_setup_webrtc_routes_passes_configured_ice_servers_to_handler(self):
        """Configured STUN and TURN servers must reach the bot's peer connection."""
        app = FastAPI()
        args = argparse.Namespace(
            folder=None,
            esp32=False,
            host="localhost",
            ice_servers=[
                {"urls": "stun:stun.l.google.com:19302"},
                {
                    "urls": "turn:turn.example.com:3478",
                    "username": "user",
                    "credential": "secret",
                },
            ],
        )
        handler_kwargs = []

        fake_modules = _fake_smallwebrtc_modules(handler_kwargs)
        ice_server_cls = fake_modules["pipecat.transports.smallwebrtc.connection"].IceServer

        with (
            patch("pipecat.runner.run._transport_routes_enabled", return_value=True),
            patch.dict(sys.modules, fake_modules),
        ):
            _setup_webrtc_routes(app, args, {})

        self.assertEqual(
            handler_kwargs[0]["ice_servers"],
            [
                ice_server_cls(urls="stun:stun.l.google.com:19302"),
                ice_server_cls(
                    urls="turn:turn.example.com:3478", username="user", credential="secret"
                ),
            ],
        )

    def test_setup_websocket_routes_skips_when_websocket_is_missing(self):
        """Plain WebSocket routes should be optional."""
        app = FastAPI()
        args = argparse.Namespace()

        with patch("pipecat.runner.run._transport_routes_enabled", return_value=False):
            _setup_websocket_routes(app, args, set())

        paths = {route.path for route in app.routes}
        self.assertNotIn("/ws-client", paths)

    def test_setup_websocket_routes_registers_when_websocket_is_available(self):
        """Plain WebSocket route should be registered when dependencies are available."""
        app = FastAPI()
        args = argparse.Namespace()

        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            _setup_websocket_routes(app, args, set())

        paths = {route.path for route in app.routes}
        self.assertIn("/ws-client", paths)

    def test_setup_telephony_routes_skips_when_websocket_is_missing(self):
        """Telephony WebSocket routes should be optional."""
        app = FastAPI()
        args = argparse.Namespace(transport=None)

        with patch("pipecat.runner.run._transport_routes_enabled", return_value=False):
            _setup_telephony_routes(app, args, set())

        paths = {route.path for route in app.routes}
        self.assertNotIn("/ws", paths)

    def test_setup_telephony_routes_registers_when_websocket_is_available(self):
        """Telephony WebSocket route should be registered when dependencies are available."""
        app = FastAPI()
        args = argparse.Namespace(transport=None)

        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            _setup_telephony_routes(app, args, set())

        paths = {route.path for route in app.routes}
        self.assertIn("/ws", paths)

    def test_setup_telephony_routes_registers_provider_webhook_for_selected_transport(self):
        """Provider webhook route should be registered for selected telephony transports."""
        app = FastAPI()
        args = argparse.Namespace(transport="twilio", proxy="example.ngrok.io")

        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            _setup_telephony_routes(app, args, set())

        post_root_routes = [
            route for route in app.routes if route.path == "/" and "POST" in route.methods
        ]
        self.assertEqual(len(post_root_routes), 1)

    def test_setup_daily_routes_skips_when_daily_is_missing(self):
        """Daily routes should be optional."""
        app = FastAPI()
        args = argparse.Namespace(dialin=False)

        with patch("pipecat.runner.run._transport_routes_enabled", return_value=False):
            _setup_daily_routes(app, args)

        paths = {route.path for route in app.routes}
        self.assertNotIn("/daily", paths)

    def test_setup_daily_routes_registers_when_daily_is_available(self):
        """Daily route should be registered when dependencies are available."""
        app = FastAPI()
        args = argparse.Namespace(dialin=False)

        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            _setup_daily_routes(app, args)

        paths = {route.path for route in app.routes}
        self.assertIn("/daily", paths)

    def test_setup_daily_routes_registers_dialin_route_when_enabled(self):
        """Daily dial-in route should be registered when requested and available."""
        app = FastAPI()
        args = argparse.Namespace(dialin=True)

        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            _setup_daily_routes(app, args)

        paths = {route.path for route in app.routes}
        self.assertIn("/daily", paths)
        self.assertIn("/daily-dialin-webhook", paths)

    def test_websocket_routes_require_fastapi_and_websockets(self):
        with patch(
            "pipecat.runner.run._is_module_available",
            side_effect=lambda module: module == "fastapi",
        ) as is_module_available:
            self.assertFalse(_transport_routes_enabled("websocket"))

        self.assertEqual(
            [call.args[0] for call in is_module_available.call_args_list],
            ["fastapi", "websockets"],
        )

    def test_start_rejects_disabled_transport_before_running_bot(self):
        app = FastAPI()
        args = argparse.Namespace(transport=None, ice_servers=[])
        _setup_unified_start_route(app, args, {})

        with patch("pipecat.runner.run._transport_routes_enabled", return_value=False):
            response = TestClient(app).post("/start", json={"transport": "daily"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"],
            (
                "Transport 'daily' is disabled in this runner environment. "
                "Check the startup banner for enabled transports."
            ),
        )

    def test_startup_message_all_transports_shows_open_url_and_transport_status(self):
        args = argparse.Namespace(
            transport=None, host="localhost", port=7860, ws_auth="none", allowed_origins=[]
        )

        def routes_enabled(transport: str) -> bool:
            return transport in {"telephony", "websocket"}

        with patch("pipecat.runner.run._transport_routes_enabled", side_effect=routes_enabled):
            output = self._capture_startup_message(args)

        self.assertEqual(
            output,
            (
                "\n"
                "🚀 Bot ready!\n"
                "   → Open: http://localhost:7860\n"
                "   → Enabled transports: telephony, websocket\n"
                "   → Disabled transports: daily (install pipecat-ai[daily]), "
                "livekit (install pipecat-ai[livekit]), "
                "webrtc (install pipecat-ai[webrtc]), "
                "moq (install pipecat-ai[moq])\n"
                "   → Allowed origins: all (no restriction)\n"
                "\n"
            ),
        )

    def test_startup_message_all_transports_omits_disabled_status_when_all_enabled(self):
        args = argparse.Namespace(
            transport=None, host="localhost", port=7860, ws_auth="none", allowed_origins=[]
        )

        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            output = self._capture_startup_message(args)

        self.assertEqual(
            output,
            (
                "\n"
                "🚀 Bot ready!\n"
                "   → Open: http://localhost:7860\n"
                "   → Enabled transports: daily, livekit, webrtc, telephony, websocket, moq\n"
                "   → Allowed origins: all (no restriction)\n"
                "\n"
            ),
        )

    def test_startup_message_webrtc_uses_root_open_url(self):
        args = argparse.Namespace(
            transport="webrtc", host="localhost", port=7860, esp32=False, whatsapp=False
        )

        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            output = self._capture_startup_message(args)

        self.assertIn("   → Open: http://localhost:7860\n", output)
        self.assertNotIn("/client", output)

    def test_startup_message_daily_uses_root_open_url(self):
        args = argparse.Namespace(transport="daily", host="localhost", port=7860, dialin=False)

        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            output = self._capture_startup_message(args)

        self.assertIn("   → Open: http://localhost:7860\n", output)
        self.assertNotIn("/daily in your browser", output)

    def test_startup_message_telephony_keeps_provider_endpoint_details(self):
        args = argparse.Namespace(
            transport="twilio",
            host="localhost",
            port=7860,
            proxy="example.ngrok.io",
            ws_auth="none",
            allowed_origins=[],
        )

        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            output = self._capture_startup_message(args)

        self.assertIn("   → Open: http://localhost:7860\n", output)
        self.assertIn("   → XML webhook: http://localhost:7860/\n", output)
        self.assertIn("   → WebSocket:   ws://localhost:7860/ws\n", output)


@unittest.skipUnless(LIVEKIT_AVAILABLE, "livekit package not installed")
class TestLiveKitRunnerRoutes(unittest.TestCase):
    def test_livekit_credentials_raises_when_unconfigured(self):
        from pipecat.runner.livekit import livekit_credentials

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(Exception, "LIVEKIT_URL"):
                livekit_credentials()

    def test_livekit_credentials_returns_configured_values(self):
        from pipecat.runner.livekit import livekit_credentials

        env = {
            "LIVEKIT_URL": "wss://test.livekit.cloud",
            "LIVEKIT_API_KEY": "key",
            "LIVEKIT_API_SECRET": "secret",
        }
        with patch.dict("os.environ", env, clear=True):
            url, api_key, api_secret = livekit_credentials()
        self.assertEqual((url, api_key, api_secret), (env["LIVEKIT_URL"], "key", "secret"))

    def test_generate_session_tokens_suffixes_identity_with_session_id(self):
        """Regression: fixed identities collide across concurrent sessions sharing
        a room (e.g. a fixed LIVEKIT_ROOM_NAME), evicting the earlier participant.
        """
        import uuid

        import jwt

        from pipecat.runner.livekit import generate_session_tokens

        agent_token_1, user_token_1 = generate_session_tokens(
            "shared-room", str(uuid.uuid4()), "key", "secret"
        )
        agent_token_2, user_token_2 = generate_session_tokens(
            "shared-room", str(uuid.uuid4()), "key", "secret"
        )

        def identity(token: str) -> str:
            return jwt.decode(token, options={"verify_signature": False})["sub"]

        identities = {
            identity(agent_token_1),
            identity(user_token_1),
            identity(agent_token_2),
            identity(user_token_2),
        }
        self.assertEqual(
            len(identities), 4, "every session/role pair should have a unique identity"
        )


class TestWsAuthTokens(unittest.TestCase):
    """Unit tests for the HMAC WebSocket session token helpers."""

    # --- _generate_ws_token ---

    def test_generate_token_has_two_parts(self):
        token = _generate_ws_token()
        parts = token.split(".")
        self.assertEqual(len(parts), 2, "Token must be <payload>.<signature>")

    def test_generate_token_is_unique(self):
        self.assertNotEqual(_generate_ws_token(), _generate_ws_token())

    # --- _verify_and_consume_ws_token ---

    def test_validate_accepts_fresh_token(self):
        token = _generate_ws_token()
        self.assertTrue(_verify_and_consume_ws_token(set(), token))

    def test_validate_consumes_token_on_success(self):
        used: set[str] = set()
        token = _generate_ws_token()
        _verify_and_consume_ws_token(used, token)
        self.assertIn(token, used)

    def test_validate_rejects_replayed_token(self):
        used: set[str] = set()
        token = _generate_ws_token()
        self.assertTrue(_verify_and_consume_ws_token(used, token))
        self.assertFalse(_verify_and_consume_ws_token(used, token))

    def test_validate_rejects_expired_token(self):
        with patch("pipecat.runner.run.time") as mock_time:
            mock_time.time.return_value = 1_000_000_000
            token = _generate_ws_token(ttl=300)
            mock_time.time.return_value = 1_000_000_000 + 301
            self.assertFalse(_verify_and_consume_ws_token(set(), token))

    def test_validate_rejects_tampered_signature(self):
        token = _generate_ws_token()
        payload, _ = token.rsplit(".", 1)
        self.assertFalse(_verify_and_consume_ws_token(set(), f"{payload}.badsignature"))

    def test_validate_rejects_tampered_payload(self):
        import base64
        import json

        _, sig = _generate_ws_token().rsplit(".", 1)
        bad_payload = (
            base64.urlsafe_b64encode(json.dumps({"exp": 9_999_999_999}).encode())
            .decode()
            .rstrip("=")
        )
        self.assertFalse(_verify_and_consume_ws_token(set(), f"{bad_payload}.{sig}"))

    def test_validate_rejects_malformed_token_no_dot(self):
        self.assertFalse(_verify_and_consume_ws_token(set(), "nodothere"))

    def test_validate_rejects_malformed_token_bad_base64(self):
        self.assertFalse(_verify_and_consume_ws_token(set(), "!!!.invalidsig"))

    # --- _extract_ws_token ---

    def test_extract_reads_authorization_bearer_header(self):
        ws = MagicMock()
        ws.headers.get.return_value = "Bearer mytoken"
        ws.query_params.get.return_value = None
        self.assertEqual(_extract_ws_token(ws), "mytoken")

    def test_extract_bearer_is_case_insensitive(self):
        ws = MagicMock()
        ws.headers.get.return_value = "BEARER mytoken"
        ws.query_params.get.return_value = None
        self.assertEqual(_extract_ws_token(ws), "mytoken")

    def test_extract_falls_back_to_query_param(self):
        ws = MagicMock()
        ws.headers.get.return_value = ""
        ws.query_params.get.return_value = "qptoken"
        self.assertEqual(_extract_ws_token(ws), "qptoken")

    def test_extract_prefers_header_over_query_param(self):
        ws = MagicMock()
        ws.headers.get.return_value = "Bearer headertoken"
        ws.query_params.get.return_value = "qptoken"
        self.assertEqual(_extract_ws_token(ws), "headertoken")

    def test_extract_returns_none_when_absent(self):
        ws = MagicMock()
        ws.headers.get.return_value = ""
        ws.query_params.get.return_value = None
        self.assertIsNone(_extract_ws_token(ws))


class TestWsAuthRouteRegistration(unittest.TestCase):
    """Route registration tests for path-token WebSocket variants."""

    def test_websocket_routes_register_path_token_route(self):
        app = FastAPI()
        args = argparse.Namespace(ws_auth="token")

        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            _setup_websocket_routes(app, args, set())

        paths = {route.path for route in app.routes}
        self.assertIn("/ws-client", paths)
        self.assertIn("/ws-client/{token}", paths)

    def test_telephony_routes_register_path_token_route(self):
        app = FastAPI()
        args = argparse.Namespace(transport=None, ws_auth="token")

        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            _setup_telephony_routes(app, args, set())

        paths = {route.path for route in app.routes}
        self.assertIn("/ws", paths)
        self.assertIn("/ws/{token}", paths)


class TestWsAuthStartEndpoint(unittest.TestCase):
    """Tests for /start returning HMAC tokens when ws_auth='token'."""

    def _make_app(self, ws_auth: str) -> FastAPI:
        app = FastAPI()
        args = argparse.Namespace(
            transport=None,
            ws_auth=ws_auth,
            host="localhost",
            port=7860,
            ice_servers=[],
        )
        _setup_unified_start_route(app, args, {})
        return app

    def test_start_websocket_returns_none_when_auth_disabled(self):
        app = self._make_app(ws_auth="none")
        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            response = TestClient(app).post("/start", json={"transport": "websocket"})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json().get("token"))

    def test_start_websocket_returns_real_token_when_auth_enabled(self):
        app = self._make_app(ws_auth="token")
        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            response = TestClient(app).post("/start", json={"transport": "websocket"})
        self.assertEqual(response.status_code, 200)
        token = response.json()["token"]
        self.assertNotEqual(token, None)
        # Token must be a valid, fresh HMAC token
        self.assertTrue(_verify_and_consume_ws_token(set(), token))

    def test_start_telephony_omits_token_when_auth_disabled(self):
        app = self._make_app(ws_auth="none")
        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            response = TestClient(app).post("/start", json={"transport": "twilio"})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("token", response.json())

    def test_start_telephony_returns_token_when_auth_enabled(self):
        app = self._make_app(ws_auth="token")
        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            response = TestClient(app).post("/start", json={"transport": "twilio"})
        self.assertEqual(response.status_code, 200)
        token = response.json().get("token")
        self.assertIsNotNone(token)
        self.assertTrue(_verify_and_consume_ws_token(set(), token))


class TestWsAuthConnectionBehavior(unittest.TestCase):
    """WebSocket connection tests verifying auth enforcement at the ASGI layer."""

    def _make_ws_client_app(self, ws_auth: str) -> tuple[FastAPI, set]:
        app = FastAPI()
        args = argparse.Namespace(ws_auth=ws_auth, allowed_origins=[])
        used: set[str] = set()
        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            _setup_websocket_routes(app, args, used)
        return app, used

    def _make_telephony_app(self, ws_auth: str) -> tuple[FastAPI, set]:
        app = FastAPI()
        args = argparse.Namespace(ws_auth=ws_auth, transport=None, allowed_origins=[])
        used: set[str] = set()
        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            _setup_telephony_routes(app, args, used)
        return app, used

    # /ws-client (plain WebSocket)

    def test_plain_ws_rejects_without_token_when_auth_enabled(self):
        app, _ = self._make_ws_client_app(ws_auth="token")
        with self.assertRaises(WebSocketDisconnect) as cm:
            with TestClient(app).websocket_connect("/ws-client"):
                pass
        self.assertEqual(cm.exception.code, 4003)

    def test_plain_ws_rejects_invalid_token_in_query_param(self):
        app, _ = self._make_ws_client_app(ws_auth="token")
        with self.assertRaises(WebSocketDisconnect) as cm:
            with TestClient(app).websocket_connect("/ws-client?token=badtoken"):
                pass
        self.assertEqual(cm.exception.code, 4003)

    def test_plain_ws_accepts_valid_token_in_query_param(self):
        app, _ = self._make_ws_client_app(ws_auth="token")
        token = _generate_ws_token()
        with patch("pipecat.runner.run._run_websocket_bot", new=AsyncMock()):
            with TestClient(app).websocket_connect(f"/ws-client?token={token}"):
                pass  # connection accepted; bot mock returns immediately

    def test_plain_ws_accepts_valid_token_in_path(self):
        app, _ = self._make_ws_client_app(ws_auth="token")
        token = _generate_ws_token()
        with patch("pipecat.runner.run._run_websocket_bot", new=AsyncMock()):
            with TestClient(app).websocket_connect(f"/ws-client/{token}"):
                pass

    def test_plain_ws_rejects_replayed_token(self):
        app, used = self._make_ws_client_app(ws_auth="token")
        token = _generate_ws_token()
        used.add(token)  # mark already consumed
        with self.assertRaises(WebSocketDisconnect) as cm:
            with TestClient(app).websocket_connect(f"/ws-client?token={token}"):
                pass
        self.assertEqual(cm.exception.code, 4003)

    def test_plain_ws_allows_any_connection_when_auth_disabled(self):
        app, _ = self._make_ws_client_app(ws_auth="none")
        with patch("pipecat.runner.run._run_websocket_bot", new=AsyncMock()):
            with TestClient(app).websocket_connect("/ws-client"):
                pass

    # /ws (telephony WebSocket)

    def test_telephony_ws_rejects_without_token_when_auth_enabled(self):
        app, _ = self._make_telephony_app(ws_auth="token")
        with self.assertRaises(WebSocketDisconnect) as cm:
            with TestClient(app).websocket_connect("/ws"):
                pass
        self.assertEqual(cm.exception.code, 4003)

    def test_telephony_ws_accepts_valid_token_in_path(self):
        app, _ = self._make_telephony_app(ws_auth="token")
        token = _generate_ws_token()
        with patch("pipecat.runner.run._run_telephony_bot", new=AsyncMock()):
            with TestClient(app).websocket_connect(f"/ws/{token}"):
                pass

    def test_telephony_ws_allows_any_connection_when_auth_disabled(self):
        app, _ = self._make_telephony_app(ws_auth="none")
        with patch("pipecat.runner.run._run_telephony_bot", new=AsyncMock()):
            with TestClient(app).websocket_connect("/ws"):
                pass


class TestAllowedOriginsUtil(unittest.TestCase):
    """Unit tests for the is_origin_allowed utility."""

    def setUp(self):
        from pipecat.utils.security.allowed_origins import is_origin_allowed

        self.is_origin_allowed = is_origin_allowed

    def test_empty_list_allows_any_origin(self):
        self.assertTrue(self.is_origin_allowed("https://example.com", []))

    def test_empty_list_allows_missing_origin(self):
        self.assertTrue(self.is_origin_allowed("", []))

    def test_matching_origin_is_allowed(self):
        self.assertTrue(self.is_origin_allowed("https://example.com", ["https://example.com"]))

    def test_non_matching_origin_is_rejected(self):
        self.assertFalse(self.is_origin_allowed("https://evil.com", ["https://example.com"]))

    def test_missing_origin_is_rejected_when_origins_configured(self):
        self.assertFalse(self.is_origin_allowed("", ["https://example.com"]))

    def test_matching_is_case_insensitive(self):
        self.assertTrue(self.is_origin_allowed("https://Example.COM", ["https://example.com"]))

    def test_multiple_allowed_origins(self):
        allowed = ["https://a.com", "https://b.com"]
        self.assertTrue(self.is_origin_allowed("https://a.com", allowed))
        self.assertTrue(self.is_origin_allowed("https://b.com", allowed))
        self.assertFalse(self.is_origin_allowed("https://c.com", allowed))


class TestWsOriginConnectionBehavior(unittest.TestCase):
    """WebSocket connection tests verifying origin enforcement at the ASGI layer."""

    def _make_ws_client_app(self, allowed_origins: list) -> FastAPI:
        app = FastAPI()
        args = argparse.Namespace(ws_auth="none", allowed_origins=allowed_origins)
        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            _setup_websocket_routes(app, args, set())
        return app

    def _make_telephony_app(self, allowed_origins: list) -> FastAPI:
        app = FastAPI()
        args = argparse.Namespace(ws_auth="none", transport=None, allowed_origins=allowed_origins)
        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            _setup_telephony_routes(app, args, set())
        return app

    # /ws-client (plain WebSocket)

    def test_plain_ws_rejects_disallowed_origin(self):
        app = self._make_ws_client_app(allowed_origins=["https://allowed.com"])
        with self.assertRaises(WebSocketDisconnect) as cm:
            with TestClient(app).websocket_connect(
                "/ws-client", headers={"Origin": "https://evil.com"}
            ):
                pass
        self.assertEqual(cm.exception.code, 4003)

    def test_plain_ws_rejects_missing_origin_when_origins_configured(self):
        app = self._make_ws_client_app(allowed_origins=["https://allowed.com"])
        with self.assertRaises(WebSocketDisconnect) as cm:
            with TestClient(app).websocket_connect("/ws-client"):
                pass
        self.assertEqual(cm.exception.code, 4003)

    def test_plain_ws_accepts_allowed_origin(self):
        app = self._make_ws_client_app(allowed_origins=["https://allowed.com"])
        with patch("pipecat.runner.run._run_websocket_bot", new=AsyncMock()):
            with TestClient(app).websocket_connect(
                "/ws-client", headers={"Origin": "https://allowed.com"}
            ):
                pass

    def test_plain_ws_allows_any_origin_when_origins_not_configured(self):
        app = self._make_ws_client_app(allowed_origins=[])
        with patch("pipecat.runner.run._run_websocket_bot", new=AsyncMock()):
            with TestClient(app).websocket_connect(
                "/ws-client", headers={"Origin": "https://anyone.com"}
            ):
                pass

    # /ws (telephony WebSocket)

    def test_telephony_ws_rejects_disallowed_origin(self):
        app = self._make_telephony_app(allowed_origins=["https://allowed.com"])
        with self.assertRaises(WebSocketDisconnect) as cm:
            with TestClient(app).websocket_connect("/ws", headers={"Origin": "https://evil.com"}):
                pass
        self.assertEqual(cm.exception.code, 4003)

    def test_telephony_ws_accepts_allowed_origin(self):
        app = self._make_telephony_app(allowed_origins=["https://allowed.com"])
        with patch("pipecat.runner.run._run_telephony_bot", new=AsyncMock()):
            with TestClient(app).websocket_connect(
                "/ws", headers={"Origin": "https://allowed.com"}
            ):
                pass


class TestIceServerParsing(unittest.TestCase):
    """Unit tests for the --ice-servers and PIPECAT_ICE_SERVERS parser."""

    def test_empty_input_yields_no_servers(self):
        self.assertEqual(_parse_ice_servers(None), [])
        self.assertEqual(_parse_ice_servers(""), [])
        self.assertEqual(_parse_ice_servers([]), [])

    def test_cli_tokens_of_bare_urls(self):
        self.assertEqual(
            _parse_ice_servers(["stun:stun.l.google.com:19302", "turn:turn.example.com:3478"]),
            [{"urls": "stun:stun.l.google.com:19302"}, {"urls": "turn:turn.example.com:3478"}],
        )

    def test_cli_token_of_json_object_keeps_credentials(self):
        self.assertEqual(
            _parse_ice_servers(
                ['{"urls": "turn:turn.example.com:3478", "username": "u", "credential": "p"}']
            ),
            [{"urls": "turn:turn.example.com:3478", "username": "u", "credential": "p"}],
        )

    def test_env_value_is_split_on_commas(self):
        self.assertEqual(
            _parse_ice_servers("stun:stun.l.google.com:19302, turn:turn.example.com:3478"),
            [{"urls": "stun:stun.l.google.com:19302"}, {"urls": "turn:turn.example.com:3478"}],
        )

    def test_env_value_may_be_a_json_array(self):
        self.assertEqual(
            _parse_ice_servers(
                '[{"urls": ["turn:turn.example.com:3478"], "username": "u", "credential": "p"},'
                ' "stun:stun.l.google.com:19302"]'
            ),
            [
                {"urls": ["turn:turn.example.com:3478"], "username": "u", "credential": "p"},
                {"urls": "stun:stun.l.google.com:19302"},
            ],
        )

    def test_null_credentials_are_dropped(self):
        self.assertEqual(
            _parse_ice_servers('{"urls": "stun:stun.example.com:3478", "username": null}'),
            [{"urls": "stun:stun.example.com:3478"}],
        )

    def test_entry_without_urls_is_rejected(self):
        with self.assertRaises(ValueError):
            _parse_ice_servers('{"username": "u", "credential": "p"}')

    def test_unsupported_key_is_rejected(self):
        with self.assertRaises(ValueError) as cm:
            _parse_ice_servers('{"urls": "turn:turn.example.com:3478", "user": "u"}')
        self.assertIn("user", str(cm.exception))

    def test_malformed_json_is_rejected(self):
        with self.assertRaises(ValueError):
            _parse_ice_servers('{"urls": ')
        with self.assertRaises(ValueError):
            _parse_ice_servers('[{"urls": "turn:turn.example.com:3478"')

    def test_json_scalar_is_rejected(self):
        with self.assertRaises(ValueError):
            _parse_ice_servers("[7]")


class TestStartIceConfig(unittest.TestCase):
    """Tests for the iceConfig the WebRTC /start response hands to clients."""

    def _post_start(self, ice_servers: list, body: dict) -> dict:
        app = FastAPI()
        args = argparse.Namespace(transport=None, ice_servers=ice_servers)
        _setup_unified_start_route(app, args, {})
        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            response = TestClient(app).post("/start", json={"transport": "webrtc", **body})
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_no_ice_config_without_configuration_or_request_flag(self):
        self.assertIsNone(self._post_start([], {}).get("iceConfig"))

    def test_request_flag_still_returns_the_public_stun_fallback(self):
        result = self._post_start([], {"enableDefaultIceServers": True})
        self.assertEqual(
            result["iceConfig"],
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )

    def test_configured_servers_are_returned_to_the_client(self):
        configured = [
            {"urls": "turn:turn.example.com:3478", "username": "u", "credential": "p"},
        ]
        result = self._post_start(configured, {})
        self.assertEqual(result["iceConfig"], {"iceServers": configured})

    def test_configured_servers_take_precedence_over_the_fallback(self):
        configured = [{"urls": "stun:stun.example.com:3478"}]
        result = self._post_start(configured, {"enableDefaultIceServers": True})
        self.assertEqual(result["iceConfig"], {"iceServers": configured})


if __name__ == "__main__":
    unittest.main()
