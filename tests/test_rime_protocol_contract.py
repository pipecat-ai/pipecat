"""Check Rime v1 encoding against fixed wire envelopes, independent of the installed schema."""

import json
from pathlib import Path
from typing import Any

import pytest
from google.protobuf import json_format
from rime_api import text_to_speech_pb2 as proto

from pipecat.services.rime.tts_v1 import RimeV1TTSService

_FIXTURES = json.loads((Path(__file__).parent / "fixtures/rime_websocket_v1.json").read_text())


@pytest.mark.parametrize("fixture", _FIXTURES)
@pytest.mark.parametrize("protocol", ["binary", "json"])
def test_fixed_wire_envelope(fixture: dict[str, Any], protocol: str) -> None:
    service = RimeV1TTSService(api_key="key", websocket_protocol=protocol)
    binary = bytes.fromhex(fixture["binary"])
    expected = fixture["json"]
    if fixture["message"] == "WebSocketRequest":
        request = json_format.ParseDict(expected, proto.WebSocketRequest())
        encoded = service._encode(request)
        assert encoded == binary if isinstance(encoded, bytes) else json.loads(encoded) == expected
        assert json_format.MessageToDict(proto.WebSocketRequest.FromString(binary)) == expected
    else:
        response = service._decode(binary if protocol == "binary" else json.dumps(expected))
        assert json_format.MessageToDict(response) == expected
        assert response.SerializeToString() == binary
