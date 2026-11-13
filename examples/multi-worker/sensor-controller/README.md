# sensor-controller

Two `PipelineWorker`s side by side, communicating only over job RPC. A voice agent has a single `ask_controller(question)` tool that forwards every temperature-related request to a worker; the worker owns a simulated thermometer and its own tool-calling LLM that decides how to answer (read the current value, inspect rolling stats, change the target, change the response rate). The worker is a plain `PipelineWorker` — it does not subclass `LLMWorker` and is not bridged.

See the [top-level multi-worker README](../README.md) for setup and shared environment variables.

## Running

```bash
uv run sensor-controller/sensor-controller.py
```

Open <http://localhost:7860/client> in your browser to talk to your bot.

To use Daily transport:

```bash
uv run sensor-controller/sensor-controller.py --transport daily
```

## Example questions

- "What's the temperature?"
- "Make it warmer."
- "Is it stable yet?"
- "Why is it slow?" / "Speed up the response."
- "What was the highest reading?"

## Architecture

```
Voice agent (transport + STT + LLM + TTS, tool: ask_controller)
  └── job → Controller (PipelineWorker)
              └── SensorReader -> SensorStats -> user_agg -> llm -> assistant_agg
```

- **[`sensor-controller.py`](sensor-controller.py)** — `build_sensor_controller()` returns a plain `PipelineWorker`. Jobs arrive via `@worker.event_handler("on_job_request")`, the question is queued onto the worker LLM, and the LLM's reply is paired back to the job via the assistant aggregator's `on_assistant_turn_stopped` event.
- **[`sensor.py`](sensor.py)** — Two custom `FrameProcessor` subclasses: `SensorReader` runs an autonomous tick loop that emits a `SensorReadingFrame` each second (first-order lag toward target plus Gaussian noise; mutable target and response rate); `SensorStats` maintains rolling min/max/avg/trend.
