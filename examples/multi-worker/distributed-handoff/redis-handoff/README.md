# redis-handoff

Same two-worker handoff as [`local-handoff`](../../local-handoff/), but each worker runs as a separate process connected via Redis pub/sub. Requires `uv add "pipecat-ai[redis]"`.

See the [top-level multi-worker README](../../README.md) for shared environment variables.

## Quick start (single machine, local Redis)

_Terminal 1_: start Redis

```bash
docker run --rm -p 6379:6379 redis:7
```

_Terminal 2_: start the greeter worker

```bash
uv run distributed-handoff/redis-handoff/llm.py greeter
```

_Terminal 3_: start the support worker

```bash
uv run distributed-handoff/redis-handoff/llm.py support
```

_Terminal 4_: start the main transport worker

```bash
uv run distributed-handoff/redis-handoff/main.py
```

All processes connect to `redis://localhost:6379` by default.

## Running across machines

Point each process at the same Redis instance:

_Machine A_

```bash
uv run distributed-handoff/redis-handoff/main.py --redis-url redis://your-redis-host:6379
```

_Machine B_

```bash
uv run distributed-handoff/redis-handoff/llm.py greeter --redis-url redis://your-redis-host:6379
```

_Machine C_

```bash
uv run distributed-handoff/redis-handoff/llm.py support --redis-url redis://your-redis-host:6379
```

## Architecture

```
Machine A                    Redis                    Machine B
+------------+          +-------------+          +-------------+
| main.py    |  <---->  | pub/sub     |  <---->  |    llm.py   |
| (transport,|          | channel:    |          |  (greeter)  |
|  STT, TTS) |          | pipecat:acme|          +-------------+
+------------+          +-------------+          +-------------+
                               ^                 |    llm.py   |
                               +-------------->  |  (support)  |
                                                 +-------------+
```

- **[main.py](main.py)** — Transport worker: Daily/WebRTC, Deepgram STT, Cartesia TTS, and a `BusBridgeProcessor` over a `RedisBus`.
- **[llm.py](llm.py)** — LLM worker: runs either `greeter` or `support` with OpenAI behind a bridged `LLMWorker`.
