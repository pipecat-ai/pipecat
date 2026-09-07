# Pipecat Flows Examples

[Pipecat Flows](../../src/pipecat/flows) is the structured-conversation framework built into Pipecat. It lets you build both predefined conversation paths and dynamically generated flows while handling the complexities of state management and LLM interactions. These examples show it in action.

## Hello, world

[`hello_world.py`](./hello_world.py) is the smallest possible Flow: a bot that asks for your favorite color and then says goodbye. It's a good first read — it shows the basics of nodes, functions, and transitions. To run it, see Setup below.

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
- [`restaurant_reservation.py`](./restaurant_reservation.py) — reservation system with availability checking
- [`patient_intake.py`](./patient_intake.py) — medical intake system showing complex state management
- [`insurance_quote.py`](./insurance_quote.py) — insurance quote system with data collection
- [`podcast_interview.py`](./podcast_interview.py) — podcast interview flow

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
