# Running MoQ direct mode locally

**TEMPORARY** — review scaffolding for the `--moq-direct` PR. Delete this file
once `pipecat-ai-prebuilt` ships a release containing direct-mode support.

`--moq-direct` drops the `/start` rendezvous: the runner watches the relay for
browsers and starts a bot for each one, and the browser is told where to meet it
through the client URL. No released `pipecat-ai-prebuilt` reads that URL yet, so
the prebuilt client has to be run from source to try this.

Two repos, no npm linking — both JS dependencies this needs are already
published.

## 0. Check out both branches

```bash
git -C ~/pipecat checkout claude/moq-client-js-python-0dff25
git -C ~/pipecat-prebuilt checkout claude/moq-direct-mode
```

## 1. Bot

Dials a public relay, so nothing needs to be reachable from outside:

```bash
cd ~/pipecat
uv run python examples/transports/transports-moq.py \
    -t moq --moq-connect https://cdn.moq.pro/anon --moq-direct
```

It prints the client URL, then waits on the relay until a browser shows up:

```
   → Open:      http://localhost:7860/client/?relay=...&ns=pipecat-<random>&botId=response&clientId=request
   → Relay:     cdn.moq.pro:443/anon
   → Namespace: pipecat-<random>
```

## 2. Prebuilt client

```bash
cd ~/pipecat-prebuilt/client
npm install
npm run dev
```

## 3. Open it

Take the URL the bot printed and swap the `http://localhost:7860/client/` prefix
for the Vite dev URL (usually `http://localhost:5173/`), keeping the query
string. Then press Connect.

**Use the Vite URL, not `http://localhost:7860/client/`** — that path serves the
published `pipecat-ai-prebuilt` wheel, which doesn't read these query params.
The transport dropdown selects "Media over QUIC" on its own, because the URL
names a MoQ session.

A bot introduces itself as soon as the browser publishes its side, and shuts
down when you disconnect. Open the same URL in a second tab or on another
machine to get a second, independent call.

## What to look for

- No `POST /start` in the network tab — the client connects the transport
  straight from the query params.
- The runner logs `client '<uuid>' arrived, starting a bot` per visitor, and
  each bot publishes `<namespace>/response/<uuid>` against the matching
  `<namespace>/request/<uuid>`. Two browsers never share a path.
- Disconnecting is a clean shutdown: no `ERROR`, no traceback.
