# Getting Started — Building Pipecat Bots with a Coding Agent

**This file is for you**: how to drive your coding agent well. The other
files here are for the **agent**: AGENTS.md teaches it how to build Pipecat
apps — scaffold with `pipecat create`, check APIs against live sources
instead of stale training data, verify its own work with headless evals —
and CLAUDE.md loads it into Claude Code.

## First: set up the Pipecat Context Hub

Your agent needs a live source of Pipecat truth — its #1 failure mode is
confidently writing outdated APIs from memory. Set up the **Pipecat Context
Hub**, a local index of Pipecat source, examples, and docs:

```bash
uvx pipecat-ai-context-hub refresh    # one-time index build; allow a few minutes
claude mcp add pipecat-context-hub -- uvx pipecat-ai-context-hub serve
```

MCP servers load at session start, so do this *before* opening the coding
session.

## Your first prompt: write a spec, not a wish

What you say first is the biggest factor in what you get back. "Build me a
phone bot" buys a long round of clarifying questions; a prompt like this
produces a working bot on the first pass. Copy it and edit:

```text
I'm building a phone assistant for my flower shop, Field & Flower, that
takes customer orders.

The bot should be able to:
  - list the available bouquets
  - check if a specific flower is in stock
  - add a flower to the order
  - get a summary of the order
  - set the delivery details
  - place the order
  - end the call

When the call starts, the bot greets the caller with exactly:
"This is Field & Flower, your local flower shop. How can I help you today?"

Services:
  - Twilio for phone calls
  - STT: Soniox
  - LLM: OpenAI
  - TTS: Cartesia
  - Deploy to Pipecat Cloud

This is a demo: use a mock backend for the flower data, and "place the
order" only needs to log the order.
```

Why this works:

- **The use case and channel are explicit** (phone assistant, Twilio), so the
  agent scaffolds the right transport instead of guessing.
- **Capabilities are a list.** Each line becomes a tool the bot can call;
  vague verbs become vague tools.
- **Wording that matters is quoted.** If a greeting, name, or phrase must be
  exact, say so verbatim — otherwise the LLM improvises.
- **Services are named.** Don't know yet? Say "help me choose" — the agent is
  instructed to confirm choices with you rather than silently pick.
- **Mock vs. real is declared**, so the agent builds exactly the backend you
  asked for and nothing more.

## During the session

- **Answer its option questions.** The agent confirms transport, services,
  and deployment before scaffolding — a quick answer beats a wrong guess.
- **Have your API keys ready.** It will stop and ask rather than invent them;
  the generated `.env.example` lists everything the bot needs.
- **Let it test.** The project scaffolds with a headless eval harness, and
  the agent verifies its own work by running scripted conversations against
  the bot. After any change, "run the evals" is a fair ask.
- **Iterate one feature at a time** once the first version works — small asks
  keep the verification loop fast.

## Learn more

Docs, quickstart, and the full API reference: https://docs.pipecat.ai
