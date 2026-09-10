# Pipecat Flows Examples

[Pipecat Flows](../../src/pipecat/flows) is the structured-conversation framework built into Pipecat: a conversation as nodes, each with its own instructions and tools, and transitions between them. A flow can be written as YAML that the bot loads at runtime, or as Python. These examples show both, and each one exists to show one thing.

## Which form to use

**Default to YAML.** The graph, each node's prompts, which tools it offers, and where each tool leads are data in a `flow.yaml`. Tools that capture data or do work are direct functions in a `handlers.py`. A function that only moves the conversation is a `transition_only` entry in the YAML with its own description and needs no code. Prompts read the manager's state with `{{ key }}`, so a value a handler stored earlier can appear in a later prompt. Because the flow is data, one deployed bot can run whichever flow a session calls for, and a non-engineer can change what the bot says or where a step leads without a deploy.

**Write the flow in Python when it needs code.** The reasons are few and specific:

- **Schema control.** A direct function's parameters come from its signature and docstring. When a tool needs an `enum`, a numeric range, or another JSON Schema constraint, define it with `FlowsFunctionSchema`.
- **Structure from runtime data.** A prompt can read state, but a node whose *shape* depends on the conversation, offering different tools or routing somewhere the graph doesn't name, has to be built in code.
- **A flow driven from outside the conversation.** Nodes set from transport events, a parallel pipeline playing hold music, another worker handing off over the bus.
- **The flow is incidental.** When the example is about a pipeline feature, the flow stays in Python beside it.

Every YAML example is a folder of three files: `bot.py` runs the pipeline, `flow.yaml` is the graph, and `handlers.py` is the Python the graph names. `hello_world` exists in both forms so the two can be read side by side.

## Setup

1. Follow the [README](../../README.md#%EF%B8%8F-developing-pipecat) steps to configure your local environment. Run the commands from the repo root.

2. Copy the [`env.example`](../../env.example) file and add API keys for the services you plan to use:

   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

3. Run any example:

   ```bash
   uv run python examples/flows/yaml/food_ordering/bot.py
   uv run python examples/flows/python/food_ordering.py
   ```

4. Open the web interface at http://localhost:7860/client/ and click "Connect".

The Python examples support multiple LLM providers (OpenAI, Anthropic, Google Gemini, AWS Bedrock) through `LLM_PROVIDER`, to demonstrate cross-provider compatibility; the YAML examples use OpenAI. Like the other Pipecat examples, they default to the SmallWebRTC transport and also support Daily (`-t daily`) and telephony providers (`-t twilio -x NGROK_HOST_NAME`) — see the [examples README](../README.md#running-examples-with-other-transports) for transport details.

## Flows as YAML

- [`yaml/hello_world/`](./yaml/hello_world) — the smallest possible flow: two nodes and one tool. Read this first.
- [`yaml/food_ordering/`](./yaml/food_ordering) — the full-featured flow: branching between pizza and sushi, a pre-action, a global function, and transition-only entries
- [`yaml/restaurant_reservation/`](./yaml/restaurant_reservation) — a branch table keyed on a tool's result, and the shim pattern for transitions that depend on logic: flow-agnostic business logic under a thin tool that reports its outcome as a `status` field
- [`yaml/patient_intake/`](./yaml/patient_intake) — data-capture tools, a branch on a boolean result, and session facts such as the practice and patient names read from state
- [`yaml/insurance_quote/`](./yaml/insurance_quote) — prompts built from computed values: the handlers store each quote in state, the results node reads it with `{{ quote.monthly_premium }}`, and adjusting the coverage re-enters the node with the new figures
- [`yaml/podcast_interview/`](./yaml/podcast_interview) — a node that transitions back to itself

## Flows in Python

- [`python/hello_world.py`](./python/hello_world.py) — the smallest possible flow, in code; the pair of `yaml/hello_world/`
- [`python/food_ordering.py`](./python/food_ordering.py) — the food-ordering flow in code, showing node and edge functions; the base for the schema variant below
- [`python/food_ordering_advanced_functionschema.py`](./python/food_ordering_advanced_functionschema.py) — the same flow with `FlowsFunctionSchema`s: an `enum` on the pizza size and type and a numeric range on the sushi count, which a direct function can only hint at in prose
- [`python/warm_transfer.py`](./python/warm_transfer.py) — a flow driven by transport events: the bot transfers the caller to a human agent, briefs the agent while the caller hears hold music from a parallel pipeline, then drops out (DailyTransport only)
- [`python/multi_worker_handoff.py`](./python/multi_worker_handoff.py) — a flow living inside a worker: a structured reservation worker hands off to and from a free-form `LLMWorker` router over the bus, sharing one conversation context
- [`python/llm_switching.py`](./python/llm_switching.py) — switching between LLM providers during a conversation; the flow is incidental to the `LLMSwitcher`

Python flows define their functions as direct functions, async functions whose schema is derived from the signature and docstring, except where the point is the schema.

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
