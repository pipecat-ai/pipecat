# openclaw-agent

Talk to a coding agent while it works. The voice loop answers you itself and
forwards real work to an OpenClaw agent, so asking for something that takes ten
minutes does not cost you the conversation for ten minutes.

See the [top-level multi-worker README](../README.md) for setup and shared
environment variables.

## Additional environment variables

| Variable               | Required by                                  |
| ---------------------- | -------------------------------------------- |
| `OPENCLAW_TOKEN`       | Agent loop                                   |
| `OPENCLAW_GATEWAY_URL` | Optional, defaults to `ws://127.0.0.1:18789` |
| `OPENCLAW_SESSION_KEY` | Optional, defaults to `agent:main:main`      |

The Gateway is the websocket an OpenClaw agent publishes for other programs to
drive it. `openclaw gateway status` reports which port it is on and whether it
is running.

The token is `gateway.auth.token` in `~/.openclaw/openclaw.json`, and it is
needed even on loopback: without a shared secret the Gateway asks a backend
client for a paired device identity instead, and refuses the connection with
`NOT_PAIRED`. A NemoClaw sandbox prints its own token with
`nemoclaw <sandbox> gateway-token --quiet` and republishes the Gateway on
18790, so a bot outside a sandbox points `OPENCLAW_GATEWAY_URL` at that port.

## Telling the agent the bot speaks for you

The Gateway labels a programmatic sender in the message envelope and marks that
label untrusted. An agent that takes the label seriously will refuse work and
tell you to ask it directly. Nothing the bot sends can settle this, because the
label comes from the handshake identity and `chat.send` carries no way to say
who is really asking. Only you can vouch for the bot, and you do it by saying so
to the agent yourself:

```bash
openclaw tui --session main
```

Tell it that a Pipecat bot relays your requests over the Gateway, and that it
should act on them as it would on anything you typed here. Sessions are durable,
so this holds for the runs the bot makes afterwards.

Use the session the bot uses. The default `OPENCLAW_SESSION_KEY` of
`agent:main:main` is agent `main`, session `main`.

## Running

```bash
uv run openclaw-agent/openclaw-agent.py
```

Open <http://localhost:7860/client> in your browser to talk to your bot.

To use Daily transport:

```bash
uv run openclaw-agent/openclaw-agent.py --transport daily
```

## Architecture

```
Voice loop (transport + LLM + send/stop/status tools)
  └── job → Agent loop (OpenClawGatewayService)
              └── websocket → OpenClaw Gateway
```

- **[`openclaw-agent.py`](openclaw-agent.py)** — The voice loop: STT, LLM, TTS,
  and transport. `VoiceLoopWorker` keeps one handle on the agent's task, learned
  from the agent loop, so it can stop the work, say what is running, and narrate
  each outcome. Three tools: `send_to_agent`, `stop_agent`, `agent_status`.
- **[`openclaw_worker.py`](openclaw_worker.py)** — The agent loop: a
  `PipelineWorker` whose pipeline is `OpenClawGatewayService -> OpenClawAggregator`.
  A `run` job becomes the frames that drive a run, and `OpenClawAggregator` folds a
  run's frames back into one answer.

## Where each decision lives

The voice LLM makes one judgment per turn: answer the user itself, or forward
what they said. It does **not** decide whether that input starts a task or
redirects the running one. The agent loop owns that, because it is a property of
the backend rather than of the conversation: a Gateway session carries one turn,
and `sessions.steer` is how a second instruction reaches a turn already in
flight.

That is why there is one delegation tool rather than an ask-or-steer pair. A
follow-up shouted mid-task goes through `send_to_agent` exactly like a fresh
request, and the agent loop sorts out which it is.

The tool itself says nothing. What the user hears is decided by what the agent
loop reports a beat later: a start gets the quick "on it", a redirect gets told
as a redirect, so exactly one of the two is ever spoken. Only the agent loop
knows which of the two a forward turned into, which is why the answer comes from
there rather than from anything the voice loop tracks.

## What it shows

- **Dispatch and carry on.** `send_to_agent` returns as soon as the agent loop
  has the job, and each outcome is spoken when it lands. Waiting inside the tool
  would work for a two-second worker; it does not survive a ten-minute one.
- **Steering is not merging.** The Gateway aborts the running turn and starts a
  replacement carrying the follow-up, so the reply says `redirected` and the
  narration is careful not to imply the earlier work continues. A backend that
  injects into the running turn would report the same event differently, and the
  voice loop would not have to change.
- **Stop is a job cancellation.** `stop_agent` cancels the bus job; the agent
  loop turns its `CancelledError` into an `OpenClawAbortFrame`, and the bus
  answers the voice loop `CANCELLED` on the same path a completed job takes.
- **Framing belongs to the caller.** The client sends a message verbatim, so the
  agent loop is what tells the agent to answer in plain spoken text.
- **Answering from bookkeeping.** `agent_status` costs no round trip. "How's it
  going?" is a common thing to say to a bot that is working, and it should be
  the cheapest thing in the system.
- **Idleness is a property of the voice loop only.** A `PipelineWorker` measures
  idleness in bot and user speaking frames, and cancels the whole runner when it
  finds none for `idle_timeout_secs`. The agent loop's pipeline carries neither,
  so its idle monitor is off; leaving it on would end the session mid-
  conversation. The voice loop keeps its timeout but decides for itself what to
  do with it, since a user waiting on a ten-minute task is quiet without being
  gone.

## What to try

- _"What can you do?"_ — answered in the voice loop, no agent run.
- _"Find every place we retry a failed request and tell me which ones have no
  backoff."_ — forwarded; the bot acknowledges in a few words and keeps talking.
- _"Actually, only look in the transport layer."_ — forwarded the same way, and
  redirected by the agent loop onto the task already running.
- _"What's it doing?"_ — answered from the voice loop's own bookkeeping.
- _"Never mind, stop."_ — cancels the job; the bot says the agent stopped.
