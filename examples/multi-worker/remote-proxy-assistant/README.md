# remote-proxy-assistant

Runs an LLM worker on a remote server, connected to the main transport worker via a WebSocket proxy. No shared bus required — the proxy workers forward bus messages point-to-point over the WebSocket.

See the [top-level multi-worker README](../README.md) for setup and shared environment variables.

## Quick start (single machine)

_Terminal 1_: start the remote assistant server

```bash
uv run remote-proxy-assistant/assistant.py
```

_Terminal 2_: start the main transport worker

```bash
uv run remote-proxy-assistant/main.py --remote-url ws://localhost:8765/ws
```

Open <http://localhost:7860/client> in your browser to talk to the bot.

## Running across machines

_Server machine_: start the assistant

```bash
uv run remote-proxy-assistant/assistant.py --host 0.0.0.0 --port 8765
```

_Client machine_: point at the server

```bash
uv run remote-proxy-assistant/main.py --remote-url ws://server-host:8765/ws
```

## Architecture

```
    +-------------+    +--------------+         +--------------+    +------------------+
    |             |    |              |         |              |    |                  |
    | Main worker |    | Proxy worker | <~~~~~> | Proxy worker |    | Assistant worker |
    |             |    |  (client)    |         |  (server)    |    |                  |
    +-------------+    +--------------+         +--------------+    +------------------+
        messages           messages                 messages             messages
            │                 │                        │                    │
  ══════════╧═════════════════╧════════         ═══════╧════════════════════╧═══════════
                Worker Bus                                       Worker Bus
  ═════════════════════════════════════         ════════════════════════════════════════
```

- **[main.py](main.py)** — Transport worker with STT, TTS, and a `BusBridge`. Spawns a `WebSocketProxyClient` that connects to the remote server and forwards `BusFrameMessage`s.
- **[assistant.py](assistant.py)** — FastAPI server. Each WebSocket connection spawns a `WebSocketProxyServer` plus a bridged `AcmeAssistant` LLM worker on a per-session `WorkerRunner`.

## Security

The proxy workers filter messages by worker name:

- Only messages targeted at the remote worker cross the WebSocket
- Only messages targeted at the local worker are accepted from the WebSocket
- Broadcast messages never cross the WebSocket

Pass HTTP headers for authentication:

```python
proxy = WebSocketProxyClient(
    "proxy",
    url="wss://server-host:8765/ws",
    remote_worker_name="assistant",
    local_worker_name="acme",
    headers={"Authorization": "Bearer <token>"},
)
```
