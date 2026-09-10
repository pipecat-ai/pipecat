# Pipecat Flows Examples

[Pipecat Flows](../../src/pipecat/flows) is the structured-conversation framework built into Pipecat. It lets you build both predefined conversation paths and dynamically generated flows while handling the complexities of state management and LLM interactions. These examples show it in action.

## Hello, world

[`hello_world.py`](./hello_world.py) is the smallest possible Flow: a bot that asks for your favorite color and then says goodbye. It's a good first read — it shows the basics of nodes, functions, and transitions. To run it, see Setup below. [`hello_world_yaml/`](./hello_world_yaml) is the same bot with its two nodes in [`flow.yaml`](./hello_world_yaml/flow.yaml) and its one tool in [`handlers.py`](./hello_world_yaml/handlers.py), the shape to copy when the flow should be configuration rather than code. Every `*_yaml/` folder has the same three files: `bot.py` runs the pipeline, `flow.yaml` is the graph, and `handlers.py` is the Python the graph names: direct functions for its tools and callables for its actions. A function that only moves the conversation to another node is written in the YAML alone, as a `transition_only` entry with its own `description`; `handlers.py` holds only the functions that capture data or do work.

## Setup

1. Follow the [README](../../README.md#%EF%B8%8F-developing-pipecat) steps to configure your local environment. Run the commands from the repo root.

2. Copy the [`env.example`](../../env.example) file and add API keys for the services you plan to use:

   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

3. Run any example:

   ```bash
   uv run python examples/flows/food_ordering.py
   ```

4. Open the web interface at http://localhost:7860/client/ and click "Connect".

All examples support multiple LLM providers (OpenAI, Anthropic, Google Gemini, AWS Bedrock) to demonstrate cross-provider compatibility. Like the other Pipecat examples, they default to the SmallWebRTC transport and also support Daily (`-t daily`) and telephony providers (`-t twilio -x NGROK_HOST_NAME`) — see the [examples README](../README.md#running-examples-with-other-transports) for transport details.

## Examples

### Core flows

- [`food_ordering.py`](./food_ordering.py) — restaurant order flow demonstrating node and edge functions
- [`food_ordering_yaml/`](./food_ordering_yaml) — the same order flow with its graph loaded from [`flow.yaml`](./food_ordering_yaml/flow.yaml) at runtime via `FlowConfig`, and its tools in [`handlers.py`](./food_ordering_yaml/handlers.py); the seam for configuring a deployed bot per session
- [`restaurant_reservation.py`](./restaurant_reservation.py) — reservation system with availability checking
- [`restaurant_reservation_yaml/`](./restaurant_reservation_yaml) — the reservation flow as [`flow.yaml`](./restaurant_reservation_yaml/flow.yaml) plus [`handlers.py`](./restaurant_reservation_yaml/handlers.py); shows the shim pattern for transitions that depend on logic: flow-agnostic business logic, a thin tool that reports its outcome as a `status` field, and a branch table in the YAML that routes on it
- [`patient_intake.py`](./patient_intake.py) — medical intake system showing complex state management
- [`patient_intake_yaml/`](./patient_intake_yaml) — the intake flow as [`flow.yaml`](./patient_intake_yaml/flow.yaml) plus [`handlers.py`](./patient_intake_yaml/handlers.py); the birthday check routes on a boolean result, and the practice and patient names come from `flow_manager.state`
- [`insurance_quote.py`](./insurance_quote.py) — insurance quote system with data collection
- [`podcast_interview.py`](./podcast_interview.py) — podcast interview flow
- [`podcast_interview_yaml/`](./podcast_interview_yaml) — the interview flow as [`flow.yaml`](./podcast_interview_yaml/flow.yaml) plus [`handlers.py`](./podcast_interview_yaml/handlers.py); its interview node transitions back to itself

### Advanced features

- [`llm_switching.py`](./llm_switching.py) — switching between LLM providers during a conversation
- [`warm_transfer.py`](./warm_transfer.py) — transferring calls between flows (DailyTransport only)
- [`multi_worker_handoff.py`](./multi_worker_handoff.py) — composing Flows with Pipecat's multi-worker framework: a structured Flows reservation worker hands off to and from a free-form `LLMWorker` router over the bus, sharing a single conversation context
- [`food_ordering_advanced_functionschema.py`](./food_ordering_advanced_functionschema.py) — the food-ordering flow defined with `FlowsFunctionSchema`s instead of direct functions, for when you need to specify a function's schema explicitly

The examples define their functions as "direct functions" — async functions whose schema is derived from the signature and docstring — which is the recommended pattern. `food_ordering_advanced_functionschema.py` shows the alternative `FlowsFunctionSchema` approach.

## Evals

Most of these examples are covered by behavioral evals that drive the bot
end-to-end and assert on which Flows functions fire and what the bot says back.
The scenarios live in [`scripts/release-evals/`](../../scripts/release-evals/)
alongside the rest of the release eval suite — see the Flows section of its
README. To run just the flows bots:

```bash
scripts/release-evals/run.sh -p flows
```

Or iterate on a single bot: run it with `-t eval`, then drive one scenario
against it with
`pipecat eval run scripts/release-evals/scenarios/<name>.yaml -v`.

## Learn more

See the [Pipecat Flows guide](https://docs.pipecat.ai/guides/features/pipecat-flows) for a full walkthrough of nodes, functions, context strategies, and actions.
