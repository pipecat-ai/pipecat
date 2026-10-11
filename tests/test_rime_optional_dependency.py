"""Check Rime services when the optional protocol package is absent."""

import subprocess
import sys


def test_http_and_legacy_services_without_rime_api() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio
import importlib.abc
import sys

class BlockRimeApi(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "rime_api" or fullname.startswith("rime_api."):
            raise ModuleNotFoundError("No module named 'rime_api'", name="rime_api")

sys.meta_path.insert(0, BlockRimeApi())

import aiohttp
from pipecat.services.rime.tts import RimeHttpTTSService, RimeNonJsonTTSService, RimeTTSService

async def check():
    async with aiohttp.ClientSession() as session:
        RimeHttpTTSService(api_key="test", aiohttp_session=session)
        RimeNonJsonTTSService(api_key="test")
        RimeTTSService(api_key="test")
        for protocol in ("binary", "json"):
            try:
                RimeTTSService(
                    api_key="test",
                    websocket_url="wss://api.rime.ai/coda/ws",
                    websocket_protocol=protocol,
                )
            except ImportError as exc:
                assert 'uv add "pipecat-ai[rime]"' in str(exc)
            else:
                raise AssertionError("v1 must require the Rime extra")
    assert "rime_api" not in sys.modules

asyncio.run(check())
""",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
