#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A minimal ACP agent, for testing an ACP client against a real subprocess.

Reads newline-delimited JSON-RPC on stdin and writes it on stdout, the way a
real agent does. On ``session/prompt`` it streams a thought, a tool call, a
permission request, and a reply, then ends the turn.

Run it as ``python tests/acp_fake_agent.py``. Pass ``--die-after-session`` to
exit as soon as a session opens, which simulates an agent crashing mid-session.
"""

import json
import sys
import threading

_write_lock = threading.Lock()


def send(message):
    """Write one message to stdout."""
    with _write_lock:
        sys.stdout.write(json.dumps(message) + "\n")
        sys.stdout.flush()


def update(session_id, **params):
    """Send a session/update notification."""
    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"sessionId": session_id, **params},
        }
    )


def main():
    """Serve requests until stdin closes."""
    next_id = 1000
    pending = {}

    for line in sys.stdin:
        message = json.loads(line)
        method = message.get("method")

        if method is None:
            # A response to something we asked the client.
            if callback := pending.pop(message.get("id"), None):
                callback(message)
            continue

        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {
                        "protocolVersion": 1,
                        "agentCapabilities": {
                            "loadSession": True,
                            "promptCapabilities": {"image": False},
                        },
                    },
                }
            )
        elif method == "session/new":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {
                        "sessionId": "test-session",
                        "modes": {
                            "currentModeId": "default",
                            "availableModes": [{"id": "default", "name": "Default"}],
                        },
                    },
                }
            )
            if "--die-after-session" in sys.argv:
                return
        elif method == "session/prompt":
            session_id = message["params"]["sessionId"]
            prompt_id = message["id"]

            update(
                session_id,
                sessionUpdate="agent_thought_chunk",
                content={"type": "text", "text": "thinking"},
            )
            update(
                session_id,
                sessionUpdate="tool_call",
                toolCallId="call-1",
                title="Read worker.py",
                kind="read",
                status="pending",
            )

            next_id += 1
            request_id = next_id

            def finish(_response, session_id=session_id, prompt_id=prompt_id):
                update(
                    session_id,
                    sessionUpdate="tool_call_update",
                    toolCallId="call-1",
                    status="completed",
                )
                update(
                    session_id,
                    sessionUpdate="agent_message_chunk",
                    content={"type": "text", "text": "done"},
                )
                send({"jsonrpc": "2.0", "id": prompt_id, "result": {"stopReason": "end_turn"}})

            pending[request_id] = finish
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "session/request_permission",
                    "params": {
                        "sessionId": session_id,
                        "toolCall": {"toolCallId": "call-1", "title": "Read worker.py"},
                        "options": [
                            {"optionId": "yes", "name": "Allow", "kind": "allow_once"},
                            {"optionId": "no", "name": "Reject", "kind": "reject_once"},
                        ],
                    },
                }
            )
        elif method == "session/cancel":
            pass
        elif "id" in message:
            send({"jsonrpc": "2.0", "id": message["id"], "result": {}})


if __name__ == "__main__":
    main()
