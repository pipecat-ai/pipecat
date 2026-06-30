# Getting Started — Building Pipecat Bots with a Coding Agent

**This file is for you**: how to drive your coding agent well. The other
files here (`AGENTS.md`, `CLAUDE.md`) are the agent's guide, instructing
it how to write, run, and test Pipecat code.

## First: set up the Pipecat Context Hub

Your agent needs a live source of Pipecat truth — its #1 failure mode is
confidently writing outdated APIs from memory. Set up the **Pipecat Context
Hub**, a local index of Pipecat source, examples, and docs:

```bash
# One-time index build, using latest package; allow a few minutes
uvx pipecat-ai-context-hub@latest refresh
# Add the MCP server (use the line for your agent)
claude mcp add pipecat-context-hub -- uvx pipecat-ai-context-hub serve   # Claude Code
codex mcp add pipecat-context-hub -- uvx pipecat-ai-context-hub serve    # Codex
```

MCP servers load at session start, so do this *before* opening the coding
session.

Re-run the refresh command to index newer content — after bumping your Pipecat
version, or periodically, since Pipecat moves fast.

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
- **Iterate one feature at a time** once the first version works — small asks
  keep the verification loop fast.

## Testing your bot

A voice bot can't be eyeballed, but you don't need a live call to test it. The
project scaffolds with a **headless eval harness** — the agent verifies its own
work by running scripted conversations against the bot, so after any change
"run the evals" is a fair ask.

- **LLM judge (optional).** Some eval checks use an LLM judge. If you already
  have a local model (via [Ollama](https://ollama.com)) the agent uses it for
  free; otherwise it'll ask whether to download one (~a few GB) or reuse your
  bot's API key.
- **Talk to it yourself.** Building a phone bot? You don't need a phone — the
  agent can wire up a free, peer-to-peer browser test transport (SmallWebRTC);
  open the local page and have a voice conversation while you iterate (ask for
  it if you don't see it).

## Learn more

Docs, quickstart, and the full API reference: https://docs.pipecat.ai
