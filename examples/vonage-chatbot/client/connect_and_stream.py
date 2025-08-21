#!/usr/bin/env python3
"""
Use a Vonage (OpenTok) Video API existing session, generate a token,
and connect its audio to your Pipecat WebSocket endpoint.
"""

import argparse
import json
import os
from typing import Dict, List

from dotenv import load_dotenv
from opentok import Client  # SDK 3.x

# ---- helpers ----------------------------------------------------------------


def parse_kv_pairs(items: List[str]) -> Dict[str, str]:
    """
    Parse CLI --header/--param entries like "Key=Value" or "Key:Value".
    """
    out: Dict[str, str] = {}
    for raw in items or []:
        sep = "=" if "=" in raw else (":" if ":" in raw else None)
        if not sep:
            raise ValueError(f"Invalid header/param format: {raw!r}. Use Key=Value")
        k, v = raw.split(sep, 1)
        out[k.strip()] = v.strip()
    return out


def comma_list(s: str | None) -> List[str]:
    return [x.strip() for x in s.split(",")] if s else []


# ---- main -------------------------------------------------------------------


def main() -> None:
    load_dotenv()

    p = argparse.ArgumentParser(
        description="Create a session and connect its audio to a WebSocket (Pipecat)."
    )
    # Auth
    p.add_argument("--api-key", default=os.getenv("VONAGE_API_KEY"), required=False)
    p.add_argument("--api-secret", default=os.getenv("VONAGE_API_SECRET"), required=False)

    # Where to connect
    p.add_argument("--ws-uri", default=os.getenv("WS_URI"), help="wss://...", required=False)
    p.add_argument("--audio-rate", type=int, default=int(os.getenv("VONAGE_AUDIO_RATE", "16000")))
    p.add_argument("--bidirectional", action="store_true", default=True)

    # An existing session which needs to be connected to pipecat-ai
    p.add_argument("--session-id", default=os.getenv("VONAGE_SESSION_ID"))

    # Optional streams and headers (to pass to the WS)
    p.add_argument(
        "--streams", default=os.getenv("VONAGE_STREAMS"), help="Comma-separated stream IDs"
    )
    p.add_argument(
        "--header",
        action="append",
        help="Extra header(s) for WS, e.g. --header X-Foo=bar (repeatable)",
    )

    # Optional: choose API base. If your SDK doesn’t accept api_url, set OPENTOK_API_URL env before run.
    p.add_argument("--api-base", default=os.getenv("OPENTOK_API_URL", "https://api.opentok.com"))

    args = p.parse_args()

    # Validate inputs
    missing = [
        k
        for k, v in {
            "api-key": args.api_key,
            "api-secret": args.api_secret,
            "ws-uri": args.ws_uri,
        }.items()
        if not v
    ]
    if missing:
        raise SystemExit(f"Missing required args/env: {', '.join(missing)}")

    # Init client (SDK 3.x supports api_url kw; if yours doesn’t, remove it and use OPENTOK_API_URL env)
    try:
        ot = Client(args.api_key, args.api_secret, api_url=args.api_base)
    except TypeError:
        # Fallback for older SDKs that don't accept api_url
        ot = Client(args.api_key, args.api_secret)

    session_id = args.session_id
    print(f"Using existing session: {session_id}")

    # Token: generate a fresh one tied to this session
    token = ot.generate_token(session_id)
    print(f"Generated token: {token[:32]}...")  # don’t print full token in logs

    # Build websocket options (mirrors your Postman body)
    ws_opts = {
        "uri": args.ws_uri,
        "audioRate": args.audio_rate,
        "bidirectional": bool(args.bidirectional),
    }

    # Optional stream filtering
    stream_list = comma_list(args.streams)
    if stream_list:
        ws_opts["streams"] = stream_list

    # Optional headers passed to your WS server
    headers = parse_kv_pairs(args.header or [])
    if headers:
        ws_opts["headers"] = headers

    print("Connecting audio to WebSocket with options:")
    print(json.dumps(ws_opts, indent=2))

    # Call the Audio Connector (this is equivalent to POST /v2/project/{apiKey}/connect)
    resp = ot.connect_audio_to_websocket(session_id, token, ws_opts)

    # The SDK returns a small object/dict; print it for visibility
    try:
        print("Connect response:", json.dumps(resp, indent=2))
    except TypeError:
        # Not JSON-serializable; just repr it
        print("Connect response:", resp)

    print("\nSuccess! Your Video session should now stream audio to/from:", args.ws_uri)


if __name__ == "__main__":
    main()
