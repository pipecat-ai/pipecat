# LLM with backend

A conversational **frontend** holds the conversation with a fast model and no tools of its own. Anything that needs tools, current information or careful reasoning it hands to a **backend**: a `BackendLLMWorker` running a heavier model with the tools. `LLMWithBackend` wraps the frontend so the pair drops into a pipeline where an LLM goes, installs the tools that join them, and runs the backend as a worker of its own.

```python
llm = LLMWithBackend(
    frontend=OpenAIResponsesLLMService(...),
    backend=BackendLLMWorker(
        llm=AnthropicLLMService(...),
        context=LLMContext(tools=[search_codebase, read_file, apply_patch, run_tests]),
    ),
)
```

The two exchange messages. The frontend's `delegate` tool puts a message to the backend and returns at once, so the frontend can acknowledge while the backend works. Everything the backend writes comes back as a message in the frontend's conversation marked `Backend:`, which the frontend relays in its own words. What the backend writes in a turn with no tool calls is a result or a question and is spoken; what it writes beside tool calls is notes on the work and is not, and a built-in `report_result` tool lets it tell the user something while it goes on working. What it is doing and thinking is recorded in the conversation too, marked `Backend (working):` and `Backend (thinking):`, silently, so the frontend can say how the work is going if asked. A second request joins the work in progress, a correction changes it, and the `cancel_delegated_work` tool stops it. The `BackendConnector` owns the session with the backend and how a request is worded: a text frontend hands over the conversation since the previous delegation (`TranscriptBackendRequestStrategy`), a speech-to-speech frontend words the request itself (`ExplicitBackendRequestStrategy`), since its context lags the audio.

## Examples

One engineering assistant, five frontends. [`backend.py`](backend.py) holds the backend, its tools and both prompts: a code change that loops through searching, reading, patching and running the tests until they pass, a two-step research task, and quick CI and pull-request lookups, each taking about as long as the real thing would.

| Example                                                        | Frontend                                              |
| -------------------------------------------------------------- | ----------------------------------------------------- |
| [`openai-responses-frontend.py`](openai-responses-frontend.py) | A cascade pipeline: STT, OpenAI's Responses API, TTS. |
| [`google-frontend.py`](google-frontend.py)                     | A cascade pipeline with a Gemini frontend.            |
| [`anthropic-frontend.py`](anthropic-frontend.py)               | A cascade pipeline with a Claude frontend.            |
| [`openai-realtime-frontend.py`](openai-realtime-frontend.py)   | OpenAI Realtime, speech to speech.                    |
| [`gemini-live-frontend.py`](gemini-live-frontend.py)           | Gemini Live, speech to speech.                        |

Run one the usual way, then connect a client and try "fix the flaky retry test in the HTTP client", then ask for something else while it works:

```bash
python openai-responses-frontend.py
```

The behavioral evals for these bots are in [`evals/llm-with-backend/`](../../../evals/llm-with-backend/).

## A backend in another process

The service addresses the backend by name over the bus, so it need not run in the same process. Give `LLMWithBackend` the worker's name instead of the worker, and run the backend under its own `WorkerRunner` on a shared network bus. See [`distributed-handoff`](../distributed-handoff/) for the bus setup.

Backend process:

```python
bus = RedisBus(redis=Redis.from_url(REDIS_URL), channel="pipecat:llm-with-backend")

backend = BackendLLMWorker(
    name="backend",
    llm=AnthropicLLMService(...),
    context=LLMContext(tools=[...]),
)

runner = WorkerRunner(bus=bus, handle_sigint=True)
await runner.add_workers(backend)
await runner.run()
```

Frontend process:

```python
bus = RedisBus(redis=Redis.from_url(REDIS_URL), channel="pipecat:llm-with-backend")

llm = LLMWithBackend(
    frontend=OpenAIResponsesLLMService(...),
    backend="backend",  # registered in the other process
)

runner = WorkerRunner(bus=bus, handle_sigint=runner_args.handle_sigint)
await runner.add_workers(worker)
await runner.run()
```

The frontend attaches to the backend when its pipeline starts and waits for the registry to report the backend ready, so a backend that starts a little late is fine. A `PgmqBus` works the same way.
