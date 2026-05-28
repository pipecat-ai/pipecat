# Pipecat Multi-Worker Examples

Build bots that are many agents working together — handing off, fanning out in parallel, delegating to long-running sidecars, or running across machines. Each example here is a complete, runnable pattern.

## Setup

From the repo root:

```bash
uv sync --all-extras
source .venv/bin/activate
cd examples/multi-worker
```

Copy the env template and fill in your API keys:

```bash
cp env.example .env
```

## Environment variables

| Variable           | Required by                             |
| ------------------ | --------------------------------------- |
| `OPENAI_API_KEY`   | LLM workers                             |
| `DEEPGRAM_API_KEY` | STT                                     |
| `CARTESIA_API_KEY` | TTS                                     |
| `DAILY_API_KEY`    | Optional: only with `--transport daily` |

Some examples need additional variables (e.g. `ANTHROPIC_API_KEY`, `DATABASE_URL`); each example's README lists them.

## Multi-worker patterns

A Pipecat **worker** is a unit of work attached to a shared bus. Workers exchange messages (lifecycle events, frame transport, job RPC) and run side-by-side. The main shapes you'll see:

- **Handoff** — Two or more LLM workers swap "who is talking". A single transport pipeline routes audio to whichever child is currently active.
- **Parallel fan-out** — One worker dispatches a job to several peers in parallel and waits for all responses. Useful when you want multiple perspectives synthesized into one reply.
- **Sidecar workers** — A main pipeline talks to a long-lived peer worker (a code agent, a hardware controller, a research bot). The peer owns its own state and the conversation stays focused.
- **Distributed bus** — Same patterns, but workers run in separate processes (or machines) connected by a network bus like Redis or PGMQ.
- **Point-to-point proxy** — Two proxy workers forward messages between a local worker and a remote one over WebSocket. No shared bus required.
- **UI workers** — A worker bridges the bus to a web client: snapshots, state updates, async job-group progress, and user actions all flow through it.

## Examples

### Local (single process, in-memory bus)

| Example                                       | What it shows                                                                     |
| --------------------------------------------- | --------------------------------------------------------------------------------- |
| [`local-handoff/`](local-handoff/)            | Two LLM workers (greeter + support) that hand off control during a conversation.  |
| [`parallel-debate/`](parallel-debate/)        | Three `LLMContextWorker`s (advocate / critic / analyst) fan out via `job_group`.  |
| [`code-assistant/`](code-assistant/)          | Voice access to your codebase via a Claude Agent SDK worker behind `job(...)`.    |
| [`sensor-controller/`](sensor-controller/)    | Voice agent forwards questions to a sidecar `PipelineWorker` owning a simulated sensor. |

### Distributed (separate processes, network bus)

| Example                                                                              | What it shows                                                      |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| [`distributed-handoff/redis-handoff/`](distributed-handoff/redis-handoff/)           | Local-handoff split across processes over a `RedisBus`.            |
| [`distributed-handoff/pgmq-handoff/`](distributed-handoff/pgmq-handoff/)             | Same shape on a `PgmqBus` (Postgres / Supabase) for ops-friendly infra. |
| [`remote-proxy-assistant/`](remote-proxy-assistant/)                                 | `WebSocketProxyClient`/`Server` connecting a local transport to a remote LLM worker. No shared bus. |

### UI workers (`UIWorker` + RTVI web client)

| Example                                                          | What it shows                                                                       |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| [`ui-worker/hello-snapshot/`](ui-worker/hello-snapshot/)         | Smallest possible UIWorker example: voice grounded in whatever's on the page.       |
| [`ui-worker/shopping-list/`](ui-worker/shopping-list/)           | Every voice turn drives the UI; speech is the input modality, the screen is truth.  |
| [`ui-worker/form-fill/`](ui-worker/form-fill/)                   | Accessibility-first voice-guided form walkthrough.                                  |
| [`ui-worker/deixis/`](ui-worker/deixis/)                         | Worker reads the user's current selection from the snapshot ("explain this").       |
| [`ui-worker/async-tasks/`](ui-worker/async-tasks/)               | `ui_job_group` fans out long-running work, streaming progress + cancellation to UI. |
| [`ui-worker/document-review/`](ui-worker/document-review/)       | Synthesis demo: snapshot + deixis + form-fill actions + async job groups in one app. |
