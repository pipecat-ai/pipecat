#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from pipecat.runner.run import (
    _setup_daily_routes,
    _setup_telephony_routes,
    _setup_unified_start_route,
    _setup_webrtc_routes,
    _setup_websocket_routes,
    _transport_route_dependencies,
    _transport_routes_enabled,
)


class TestRunnerRun(unittest.TestCase):
    def test_transport_route_dependencies_maps_transports_to_modules(self):
        self.assertEqual(_transport_route_dependencies("daily"), ("daily",))
        self.assertEqual(_transport_route_dependencies("webrtc"), ("aiortc",))
        self.assertEqual(_transport_route_dependencies("websocket"), ("fastapi", "websockets"))
        self.assertEqual(_transport_route_dependencies("twilio"), ("fastapi", "websockets"))
        self.assertEqual(_transport_route_dependencies("telnyx"), ("fastapi", "websockets"))
        self.assertEqual(_transport_route_dependencies("plivo"), ("fastapi", "websockets"))
        self.assertEqual(_transport_route_dependencies("exotel"), ("fastapi", "websockets"))
        self.assertEqual(_transport_route_dependencies("vonage"), ())

    def test_transport_routes_enabled_maps_transports_to_dependency_checks(self):
        def module_available(module: str) -> bool:
            return module in {"fastapi", "websockets"}

        with patch("pipecat.runner.run._is_module_available", side_effect=module_available):
            self.assertFalse(_transport_routes_enabled("daily"))
            self.assertFalse(_transport_routes_enabled("webrtc"))
            self.assertTrue(_transport_routes_enabled("websocket"))
            self.assertTrue(_transport_routes_enabled("twilio"))
            self.assertTrue(_transport_routes_enabled("vonage"))

    def test_setup_webrtc_routes_skips_when_aiortc_is_missing(self):
        """WebRTC routes should be optional when the webrtc extra is not installed."""
        app = FastAPI()
        args = argparse.Namespace(folder=None, esp32=False, host="localhost")

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
        args = argparse.Namespace(folder=None, esp32=False, host="localhost")

        connection_module = types.ModuleType("pipecat.transports.smallwebrtc.connection")
        connection_module.SmallWebRTCConnection = MagicMock()

        request_handler_module = types.ModuleType(
            "pipecat.transports.smallwebrtc.request_handler"
        )

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

        request_handler_module.IceCandidate = IceCandidate
        request_handler_module.SmallWebRTCPatchRequest = SmallWebRTCPatchRequest
        request_handler_module.SmallWebRTCRequest = SmallWebRTCRequest

        class MockSmallWebRTCRequestHandler:
            def __init__(self, *args, **kwargs):
                pass

            async def close(self):
                pass

        request_handler_module.SmallWebRTCRequestHandler = MockSmallWebRTCRequestHandler

        with (
            patch("pipecat.runner.run._transport_routes_enabled", return_value=True),
            patch.dict(
                sys.modules,
                {
                    "pipecat.transports.smallwebrtc.connection": connection_module,
                    "pipecat.transports.smallwebrtc.request_handler": request_handler_module,
                },
            ),
        ):
            _setup_webrtc_routes(app, args, {})

        paths = {route.path for route in app.routes}
        self.assertIn("/api/offer", paths)
        self.assertIn("/files/{filename:path}", paths)

    def test_setup_websocket_routes_skips_when_websocket_is_missing(self):
        """Plain WebSocket routes should be optional."""
        app = FastAPI()
        args = argparse.Namespace()

        with patch("pipecat.runner.run._transport_routes_enabled", return_value=False):
            _setup_websocket_routes(app, args)

        paths = {route.path for route in app.routes}
        self.assertNotIn("/ws-client", paths)

    def test_setup_websocket_routes_registers_when_websocket_is_available(self):
        """Plain WebSocket route should be registered when dependencies are available."""
        app = FastAPI()
        args = argparse.Namespace()

        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            _setup_websocket_routes(app, args)

        paths = {route.path for route in app.routes}
        self.assertIn("/ws-client", paths)

    def test_setup_telephony_routes_skips_when_websocket_is_missing(self):
        """Telephony WebSocket routes should be optional."""
        app = FastAPI()
        args = argparse.Namespace(transport=None)

        with patch("pipecat.runner.run._transport_routes_enabled", return_value=False):
            _setup_telephony_routes(app, args)

        paths = {route.path for route in app.routes}
        self.assertNotIn("/ws", paths)

    def test_setup_telephony_routes_registers_when_websocket_is_available(self):
        """Telephony WebSocket route should be registered when dependencies are available."""
        app = FastAPI()
        args = argparse.Namespace(transport=None)

        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            _setup_telephony_routes(app, args)

        paths = {route.path for route in app.routes}
        self.assertIn("/ws", paths)

    def test_setup_telephony_routes_registers_provider_webhook_for_selected_transport(self):
        """Provider webhook route should be registered for selected telephony transports."""
        app = FastAPI()
        args = argparse.Namespace(transport="twilio", proxy="example.ngrok.io")

        with patch("pipecat.runner.run._transport_routes_enabled", return_value=True):
            _setup_telephony_routes(app, args)

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
        args = argparse.Namespace(transport=None)
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


if __name__ == "__main__":
    unittest.main()
