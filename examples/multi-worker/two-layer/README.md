# Two-layer LLM

A conversational **frontend** holds the conversation with a fast model and no tools of its own. Anything that needs tools, current information or careful reasoning it hands to a **backend**: a `BackendLLMWorker` running a heavier model with the tools. `TwoLayerLLMService` wraps the frontend so the pair drops into a pipeline where an LLM goes, installs the `delegate` tool that joins them, and runs the backend as a worker of its own.

```python
llm = TwoLayerLLMService(
    frontend=OpenAILLMService(...),
    backend=BackendLLMWorker(
        llm=AnthropicLLMService(...),
        context=LLMContext(tools=[get_current_weather, get_restaurant_recommendation]),
    ),
)
```

How a delegation crosses is the `BackendConnector`'s business, built from two strategies the service picks by frontend kind unless told otherwise:

|                    | request (frontend → backend)                                        | reply (backend → frontend)                                   |
| ------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------ |
| text frontend      | `TranscriptBackendRequestStrategy`: the conversation since the previous delegation | `StrictSpeechFlagBackendReplyStrategy`: progress relayed, spoken as the backend's flag says |
| realtime frontend  | `ExplicitBackendRequestStrategy`: a request the model words itself   | `FinalOnlyBackendReplyStrategy`: the answer only             |

## Examples

| Example                                            | What it shows                                                                                          |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| [`cascade-frontend.py`](cascade-frontend.py)       | A cascade pipeline (STT + GPT + TTS) as the frontend, all defaults.                                    |
| [`realtime-frontend.py`](realtime-frontend.py)     | OpenAI Realtime as the frontend, all defaults. Same backend, same prompts as the cascade example.       |
| [`advisory-speech-flag.py`](advisory-speech-flag.py) | `AdvisorySpeechFlagBackendReplyStrategy`: the frontend hears every piece of progress and decides what to say. |

Run any of them the usual way, then connect a client:

```bash
python cascade-frontend.py
```

Or drive one with a behavioral eval:

```bash
python cascade-frontend.py -t eval --port 7860
pipecat eval run ../../../scripts/release-evals/scenarios/scripted/weather_function_call_audio.yaml --bot-url ws://localhost:7860 -v
```

## A backend in another process

The service addresses the backend by name over the bus, so it need not run in the same process. Give `TwoLayerLLMService` the worker's name instead of the worker, and run the backend under its own `WorkerRunner` on a shared network bus. See [`distributed-handoff`](../distributed-handoff/) for the bus setup.

Backend process:

```python
bus = RedisBus(redis=Redis.from_url(REDIS_URL), channel="pipecat:two-layer")

backend = BackendLLMWorker(
    name="backend",
    llm=AnthropicLLMService(...),
    context=LLMContext(tools=[get_current_weather, get_restaurant_recommendation]),
)

runner = WorkerRunner(bus=bus, handle_sigint=True)
await runner.add_workers(backend)
await runner.run()
```

Frontend process:

```python
bus = RedisBus(redis=Redis.from_url(REDIS_URL), channel="pipecat:two-layer")

llm = TwoLayerLLMService(
    frontend=OpenAILLMService(...),
    backend="backend",  # registered in the other process
)

runner = WorkerRunner(bus=bus, handle_sigint=runner_args.handle_sigint)
await runner.add_workers(worker)
await runner.run()
```

The first delegation waits for the registry to report the backend ready, so a backend that starts a little late is fine. A `PgmqBus` works the same way.
