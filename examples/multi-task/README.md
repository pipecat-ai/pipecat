# Pipecat Multi-Task Examples

This directory contains example bots that use the multi-task framework in `pipecat.tasks`, `pipecat.pipeline.runner` (with `spawn()`), and the `TaskBus`. Each example shows a different cooperation pattern between tasks: hand-off, parallel fan-out, remote workers, etc.

## Setup

From the repo root:

```bash
uv sync --all-extras
source .venv/bin/activate
cd examples/multi-task
```

Copy the env template and fill in your API keys:

```bash
cp env.example .env
```

## Environment variables

| Variable           | Required by                             |
| ------------------ | --------------------------------------- |
| `OPENAI_API_KEY`   | LLM tasks                               |
| `DEEPGRAM_API_KEY` | STT                                     |
| `CARTESIA_API_KEY` | TTS                                     |
| `DAILY_API_KEY`    | Optional: only with `--transport daily` |

Additional, example-specific variables are listed below.

## Table of contents

**[Local](#local)** (single process)

- [Handoff between LLM tasks](#handoff-between-llm-tasks)
- [Parallel debate](#parallel-debate)
- [Voice code assistant with Claude Agent SDK](#voice-code-assistant)
- [Sensor controller](#sensor-controller)

**[Distributed](#distributed)** (multi-process)

- [Handoff via Redis](#handoff-via-redis)
- [Handoff via PGMQ (Postgres)](#handoff-via-pgmq-postgres)
- [LLM task via WebSocket proxy](#llm-task-via-websocket-proxy)

# Local

Examples where all tasks run in the same process on an `AsyncQueueBus`.

## Handoff between LLM tasks

Two LLM tasks (greeter + support) that transfer control to each other during a voice conversation. A main task owns the transport pipeline and bridges frames to the bus.

### Running

```bash
uv run local-handoff/local-handoff-two-agents.py
```

Open <http://localhost:7860/client> in your browser to talk to your bot.

To use Daily transport:

```bash
uv run local-handoff/local-handoff-two-agents.py --transport daily
```

### Overview

- **[`local-handoff-two-agents.py`](local-handoff/local-handoff-two-agents.py)** — Two LLM tasks (greeter + support) that hand off via `activate_task(..., deactivate_self=True)`. The main task owns STT, TTS, transport, and a `BusBridgeProcessor`.
- **[`local-handoff-two-agents-tts.py`](local-handoff/local-handoff-two-agents-tts.py)** — Same shape, but each child task ships with its own `CartesiaTTSService` in a custom pipeline. The main task has no TTS — audio comes from whichever child is active over the bus.

## Parallel debate

Parallel fan-out using `task.job_group(...)`. A voice bot takes a topic from the user, kicks off three worker tasks in parallel (advocate, critic, analyst), waits for all three to respond, and synthesizes a balanced answer. Each worker keeps its own LLM context across rounds.

### Running

```bash
uv run parallel-debate/parallel-debate.py
```

Open <http://localhost:7860/client> in your browser to talk to your bot.

To use Daily transport:

```bash
uv run parallel-debate/parallel-debate.py --transport daily
```

### Architecture

```
Main task (transport + LLM + `debate` tool)
  └── job_group(advocate, critic, analyst)
        └── DebateWorker (LLMContextTask, one per role)
```

- **Main task**: transport (STT, TTS) + LLM moderator with a `debate` direct function that fans out via `task.job_group(...)`.
- **Debate workers**: `LLMContextTask`s spawned on the runner. Each keeps its own `LLMContext` across rounds and ships its completed turn back as a job response via the assistant-aggregator's `on_assistant_turn_stopped` event.

## Voice code assistant

Talk to your codebase hands-free. Ask questions about code, project structure, or file contents and get spoken answers based on actual files. The Claude Agent SDK worker navigates the filesystem using `Read`, `Bash`, `Glob`, and `Grep` tools.

### Additional environment variables

| Variable            | Required by                    |
| ------------------- | ------------------------------ |
| `ANTHROPIC_API_KEY` | Code worker (Claude Agent SDK) |
| `PROJECT_PATH`      | Optional, defaults to cwd      |

### Running

```bash
# Default: explores the current directory
uv run code-assistant/code-assistant.py

# Specify a project path
PROJECT_PATH=/path/to/your/project uv run code-assistant/code-assistant.py
```

Open <http://localhost:7860/client> in your browser to talk to your bot.

To use Daily transport:

```bash
uv run code-assistant/code-assistant.py --transport daily
```

### Example questions

- "What does the main module do?"
- "Find all TODO comments in the project"
- "How is error handling implemented?"
- "What dependencies does this project use?"
- "Explain the test structure"

### Architecture

```
Main task (transport + LLM + `ask_code` tool)
  └── job → CodeWorker (Claude Agent SDK)
```

- **`code-assistant.py`** — Main task: STT, LLM (with system prompt + `ask_code` direct function), TTS, and transport. The `ask_code` tool dispatches a job to the worker via `task.job("code_worker", payload=...)`.
- **`code_worker.py`** — `CodeWorker`: a bus-only `BaseTask` spawned on the runner. It accepts `@job`-style requests through the bus and runs them sequentially through a persistent Claude SDK session so follow-up questions share context.

## Sensor controller

Two `PipelineTask`s side by side, communicating only over job RPC. A voice agent has a single `ask_controller(question)` tool that forwards every temperature-related request to a worker; the worker owns a simulated thermometer and its own tool-calling LLM that decides how to answer (read the current value, inspect rolling stats, change the target, change the response rate). The worker is a plain `PipelineTask` — it does not subclass `LLMTask` and is not bridged.

### Running

```bash
uv run sensor-controller/sensor-controller.py
```

Open <http://localhost:7860/client> in your browser to talk to your bot.

To use Daily transport:

```bash
uv run sensor-controller/sensor-controller.py --transport daily
```

### Example questions

- "What's the temperature?"
- "Make it warmer."
- "Is it stable yet?"
- "Why is it slow?" / "Speed up the response."
- "What was the highest reading?"

### Architecture

```
Voice agent (transport + STT + LLM + TTS, tool: ask_controller)
  └── job → Controller (PipelineTask)
              └── SensorReader -> SensorStats -> user_agg -> llm -> assistant_agg
```

- **[`sensor-controller.py`](sensor-controller/sensor-controller.py)** — `build_sensor_controller()` returns a plain `PipelineTask`. Jobs arrive via `@worker.event_handler("on_job_request")`, the question is queued onto the worker LLM, and the LLM's reply is paired back to the job via the assistant aggregator's `on_assistant_turn_stopped` event.
- **[`sensor.py`](sensor-controller/sensor.py)** — Two custom `FrameProcessor` subclasses: `SensorReader` runs an autonomous tick loop that emits a `SensorReadingFrame` each second (first-order lag toward target plus Gaussian noise; mutable target and response rate); `SensorStats` maintains rolling min/max/avg/trend.

# Distributed

Examples where tasks run across separate processes or machines.

## Handoff via Redis

Same two-task handoff as the local example, but each task runs as a separate process connected via Redis pub/sub. Requires `pip install pipecat-ai[redis]`.

### Quick start (single machine, local Redis)

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

_Terminal 4_: start the main transport task

```bash
uv run distributed-handoff/redis-handoff/main.py
```

All processes connect to `redis://localhost:6379` by default.

### Running across machines

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

### Architecture

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

- **[main.py](distributed-handoff/redis-handoff/main.py)** — Transport task: Daily/WebRTC, Deepgram STT, Cartesia TTS, and a `BusBridgeProcessor` over a `RedisBus`.
- **[llm.py](distributed-handoff/redis-handoff/llm.py)** — LLM worker: runs either `greeter` or `support` with OpenAI behind a bridged `LLMTask`.

## Handoff via PGMQ (Postgres)

Same shape as the Redis handoff, but the bus is backed by [PGMQ](https://github.com/tembo-io/pgmq) on a shared Postgres database (e.g. Supabase). Requires `pip install pipecat-ai[pgmq]`.

### Additional environment variables

| Variable       | Required by                                                          |
| -------------- | -------------------------------------------------------------------- |
| `DATABASE_URL` | PostgreSQL DSN (e.g. Supabase pooled connection string)              |
| `PGMQ_CHANNEL` | Optional, channel prefix for queue names. Defaults to `pipecat_acme` |

### Quick start

_Terminal 1_: start the greeter worker

```bash
uv run distributed-handoff/pgmq-handoff/llm.py greeter --database-url $DATABASE_URL
```

_Terminal 2_: start the support worker

```bash
uv run distributed-handoff/pgmq-handoff/llm.py support --database-url $DATABASE_URL
```

_Terminal 3_: start the main transport task

```bash
uv run distributed-handoff/pgmq-handoff/main.py --database-url $DATABASE_URL
```

You can also set `DATABASE_URL` in `.env` and omit the `--database-url` flag.

### Architecture

Same as the Redis handoff above; the `RedisBus` is replaced by a `PgmqBus`, and the "pub/sub channel" is a set of PGMQ queues on the shared Postgres instance.

## LLM task via WebSocket proxy

Runs an LLM task on a remote server, connected to the main transport task via a WebSocket proxy. No shared bus required — the proxy tasks forward bus messages point-to-point over the WebSocket.

### Quick start (single machine)

_Terminal 1_: start the remote assistant server

```bash
uv run remote-proxy-assistant/assistant.py
```

_Terminal 2_: start the main transport task

```bash
uv run remote-proxy-assistant/main.py --remote-url ws://localhost:8765/ws
```

Open <http://localhost:7860/client> in your browser to talk to the bot.

### Running across machines

_Server machine_: start the assistant

```bash
uv run remote-proxy-assistant/assistant.py --host 0.0.0.0 --port 8765
```

_Client machine_: point at the server

```bash
uv run remote-proxy-assistant/main.py --remote-url ws://server-host:8765/ws
```

### Architecture

```
    +-------------+    +-------------+           +-------------+     +-----------------+
    |             |    |             |           |             |     |                 |
    |  Main task  |    | Proxy task  |  <~~~~~>  | Proxy task  |     | Assistant task  |
    |             |    |  (client)   |           |  (server)   |     |                 |
    +-------------+    +-------------+           +-------------+     +-----------------+
        messages           messages                  messages             messages
            │                 │                         │                    │
  ══════════╧═════════════════╧════════         ════════╧════════════════════╧═══════════
                Task Bus                                       Task Bus
  ═════════════════════════════════════         ═════════════════════════════════════════
```

- **[main.py](remote-proxy-assistant/main.py)** — Transport task with STT, TTS, and a `BusBridge`. Spawns a `WebSocketProxyClientTask` that connects to the remote server and forwards `BusFrameMessage`s.
- **[assistant.py](remote-proxy-assistant/assistant.py)** — FastAPI server. Each WebSocket connection spawns a `WebSocketProxyServerTask` plus a bridged `AcmeAssistant` LLM task on a per-session `PipelineRunner`.

### Security

The proxy tasks filter messages by task name:

- Only messages targeted at the remote task cross the WebSocket
- Only messages targeted at the local task are accepted from the WebSocket
- Broadcast messages never cross the WebSocket

Pass HTTP headers for authentication:

```python
proxy = WebSocketProxyClientTask(
    "proxy",
    url="wss://server-host:8765/ws",
    remote_task_name="assistant",
    local_task_name="acme",
    headers={"Authorization": "Bearer <token>"},
)
```
