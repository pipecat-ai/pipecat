# MoQ local dev setup (TEMP - not for merge)

End-to-end instructions for running the MoQ pipeline locally — a pipecat
MoQ bot (acting as its own MOQ server) and the prebuilt React client
talking to it through the locally-built `voice-ui-kit` +
`@pipecat-ai/moq-transport`.

The pipecat MoQ transport uses the [`moq-rs`](https://pypi.org/project/moq-rs/)
Python library, so the bot can bind its own UDP socket via `--moq-serve`
and mint a self-signed cert in-process. No separate `moq-relay` process,
no `openssl` cert dance. The only friction left is the `npm link` chain
for the still-unpublished MoQ packages, which `./scripts/moq-dev-setup.sh`
handles.

## The short, short version
```bash
for repo in . ../pipecat-client-web-transports ../pipecat-prebuilt ../voice-ui-kit; do
  ( cd "$repo" && git fetch origin vp-add-moq-transport \
                && git checkout vp-add-moq-transport )
done
uv sync --extra moq --extra daily --extra silero --extra deepgram \
        --extra cartesia --extra openai --extra runner
( cd ../pipecat-client-web-transports && npm install --ignore-scripts )
( cd ../pipecat-prebuilt/client       && npm install )
( cd ../voice-ui-kit \
    && npm pkg set "pnpm.overrides[@pipecat-ai/moq-transport]=link:../pipecat-client-web-transports/transports/moq-transport" \
    && pnpm install )
./scripts/moq-dev-setup.sh
```

## 0. Layout

Clone all four repos as siblings under one parent directory. The setup
script assumes this layout:

```
<parent>/
├── pipecat/                          # github.com/pipecat-ai/pipecat
├── pipecat-client-web-transports/    # github.com/pipecat-ai/pipecat-client-web-transports
├── pipecat-prebuilt/                 # github.com/pipecat-ai/pipecat-prebuilt
└── voice-ui-kit/                     # github.com/pipecat-ai/voice-ui-kit
```

All four repos live on the **`vp-add-moq-transport`** branch for this
work.

```bash
for repo in pipecat pipecat-client-web-transports pipecat-prebuilt voice-ui-kit; do
  ( cd "$repo" && git fetch origin vp-add-moq-transport \
                && git checkout vp-add-moq-transport )
done
```

Re-run this any time you suspect the branches have drifted (e.g. after
someone pushes a fix); the npm installs in §2 will pick up whatever the
checked-out tip is.

## 1. Tooling prerequisites

- Node 18+ and `npm`
- `pnpm` 10+ (the corepack shim on some Node versions has signing-key
  issues — if `pnpm install` fails with a corepack error, install it
  globally instead: `npm install -g --force pnpm@10.14.0`)
- `uv` for the Python side

## 2. Install everything

Paths assume you start from this `pipecat/` directory and the repos are
siblings:
- `/pipecat`
- `/pipecat-client-web-transports`
- `/pipecat-prebuilt`
- `/voice-ui-kit`

Order doesn't matter; no builds yet. The link chain that ties them
together happens after.

```bash
# Make sure every repo is on vp-add-moq-transport before installing.
for repo in . ../pipecat-client-web-transports ../pipecat-prebuilt ../voice-ui-kit; do
  ( cd "$repo" && git fetch origin vp-add-moq-transport \
                && git checkout vp-add-moq-transport )
done

uv sync --extra moq --extra daily --extra silero --extra deepgram \
        --extra cartesia --extra openai --extra runner
( cd ../pipecat-client-web-transports && npm install --ignore-scripts )
( cd ../pipecat-prebuilt/client       && npm install )
# voice-ui-kit declares @pipecat-ai/moq-transport as a regular dep, but
# it isn't on npm yet — register a pnpm override pointing at the local
# checkout BEFORE pnpm install, otherwise the install 404s.
( cd ../voice-ui-kit \
    && npm pkg set "pnpm.overrides[@pipecat-ai/moq-transport]=link:../pipecat-client-web-transports/transports/moq-transport" \
    && pnpm install )
```

The Python extras come from `transports-moq.py`'s pipeline: `silero` for
VAD, `deepgram` for STT, `cartesia` for TTS, `openai` for the LLM, and
`runner` for the FastAPI server that hosts `/start` + the prebuilt UI.
`moq` is the transport itself (pulls in `moq-rs` + `cryptography`).
`daily` is along for the ride — the example unconditionally imports
`DailyParams` at the top of the file so it can also run under `-t daily`,
even though we're using `-t moq`.

You'll also need API keys for those three services. Drop them in a
`.env` at the repo root (loaded automatically by the example):

```dotenv
DEEPGRAM_API_KEY=...
CARTESIA_API_KEY=...
OPENAI_API_KEY=...
```

`--ignore-scripts` on the transports repo is intentional: that workspace
runs each transport's `prepare`/build during install, and on this branch
`small-webrtc-transport`'s parcel build fails for an unrelated reason
(imports `@pipecat-ai/client-js` from `lib/` but the workspace root
doesn't declare it as a regular dep). We only need `moq-transport`'s
build, which §3's script handles directly.

A couple of things to know:

- `voice-ui-kit/package/package.json` declares `@pipecat-ai/moq-transport`
  as a regular versioned dep (peer + dev), but the package isn't on npm
  yet. The `npm pkg set` command above injects a `pnpm.overrides` entry
  in `voice-ui-kit/package.json` pointing the spec at the sibling
  `pipecat-client-web-transports/transports/moq-transport` checkout, so
  `pnpm install` resolves it locally instead of 404-ing on the registry.
  Drop the override once moq-transport ships to npm.

## 3. Run the dev setup script

From this `pipecat/` repo root:

```bash
./scripts/moq-dev-setup.sh
```

The script does three things, in order:

1. Sets up the `npm link` chain so the prebuilt client picks up the
   local moq-transport + the local voice-ui-kit, and so
   `@pipecat-ai/client-js` is a single physical instance across all
   three packages (prebuilt, voice-ui-kit, moq-transport). Without that
   dedupe, voice-ui-kit's tsc/dts step fails with `Property '_options'
   is protected but type 'Transport' is not a class derived from
   'Transport'` (the linked moq-transport drags in
   pipecat-client-web-transports's hoisted client-js@1.10.x, which
   collides with voice-ui-kit's pnpm-locked 1.8.x), and at runtime the
   React `PipecatClientProvider` context breaks across the duplicates.
2. Builds `moq-transport` + `voice-ui-kit` so their `dist/` reflects
   local source — now that the link chain is in place, tsc resolves a
   single `Transport` type and the dts build passes.
3. Prints the two terminal commands from §4.

(`pipecat-prebuilt/client` doesn't need an explicit build; Vite compiles
on demand.)

The link chain + auto-build is temporary — delete those blocks from the
script once both packages ship to npm.

## 4. Run the stack

The setup script prints these at the end; they're listed here for
reference. Two terminals.

### Terminal 1 — pipecat bot

From this `pipecat/` repo:

```bash
uv run python examples/transports/transports-moq.py \
    -t moq --moq-serve --moq-tls-generate localhost
```

`--moq-serve` makes the bot bind its own UDP socket (default `[::]:4080`);
`--moq-tls-generate localhost` tells it to mint a self-signed cert
in-process for that hostname. The fingerprint goes back to the browser via
`/start`, which pins it via WebTransport's `serverCertificateHashes`. No
separate relay process needed.

Server mode now refuses to start without explicit TLS config — pass
either `--moq-tls-generate <hostname>` (dev) or
`--moq-tls-cert /path/to/cert.pem --moq-tls-key /path/to/key.pem` (prod).
This is intentional, to avoid silently shipping a self-signed cert into a
production deploy.

For client mode (point the bot at an external relay), drop `--moq-serve`
and pass `--moq-connect https://relay.example.com:4080/moq` instead.

### Terminal 2 — prebuilt client dev server

```bash
cd ../pipecat-prebuilt/client
npm run dev
```

Open the **Vite dev server URL** (printed in this terminal — usually
<http://localhost:5173>), not the bot's `/client/` route. Vite proxies
`/start` + `/api` back to the bot at :7860, so the UI talks to the
right backend. Pick **MoQ** in the transport dropdown.

> **Important:** `http://localhost:7860/client/` serves the *published*
> `pipecat-ai-prebuilt` PyPI wheel that was bundled into the pipecat
> install — it doesn't know about MoQ. Our local edits live in the
> Vite dev server only. (Same gotcha if you build the local prebuilt
> for production and want pipecat to serve it: the runner mounts
> `pipecat_ai_prebuilt.frontend.PipecatPrebuiltUI` from
> `site-packages`, not your local checkout.)

The `ConsoleTemplate` should then drive the full UI — connect/disconnect,
transcript, mic device picker, SessionInfo showing "MoQ".

## 5. Iterating

- **Edited `moq-transport/src/*`?** Rebuild it: `cd
  pipecat-client-web-transports/transports/moq-transport && npm run build`.
  The npm link picks up the new `dist/` immediately; restart the
  prebuilt dev server (Vite caches imports).
- **Edited `voice-ui-kit/package/src/*`?** Rebuild: `pnpm -F
  @pipecat-ai/voice-ui-kit build` (or `build:watch`). Same restart
  caveat for the consumer.
- **Edited `pipecat-prebuilt/client/src/*`?** Vite hot-reloads
  automatically.
- **Edited the pipecat bot?** Restart terminal 1. The self-signed cert
  is fresh on every bot start (in-process generation), so no extra
  cleanup is needed.
- **Re-ran `pnpm install` in voice-ui-kit?** pnpm restores its own
  symlink for `@pipecat-ai/client-js`, undoing the dedupe. Re-run
  `./scripts/moq-dev-setup.sh` before the next `voice-ui-kit` build.

## 6. Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Browser console: `Failed to load transport "moq"` | moq-transport isn't linked into prebuilt. Re-run the setup script and check the "Linking …" output. |
| No **MoQ** entry in the transport dropdown | You're at `http://localhost:7860/client/`, which is the published `pipecat-ai-prebuilt` wheel served by the pipecat runner — it predates our MoQ work. Open the Vite dev server URL instead (usually `http://localhost:5173`); that's what `npm run dev` serves with your local edits. |
| Vite URL is a blank page, console says `Invalid hook call` / `Cannot read properties of null (reading 'useEffect')` | Two React instances. The linked voice-ui-kit ships its own pnpm-locked React (19.2.1) and prebuilt has its own (19.2.6 from npm); hooks break across the boundary. Fix is in `pipecat-prebuilt/client/vite.config.js` — `resolve.dedupe: ["react", "react-dom", "react/jsx-runtime", "@pipecat-ai/client-js", "@pipecat-ai/client-react"]`. Pull the latest of that file; it's already set. |
| `Property '_options' is protected` during voice-ui-kit build | Two copies of `@pipecat-ai/client-js` (1.8.x in voice-ui-kit, 1.10.x hoisted in pipecat-client-web-transports). Re-run the setup script before re-trying the build. If you re-ran `pnpm install` in voice-ui-kit after the script, run the script again — pnpm undoes the dedupe symlink. |
| `npm install` in pipecat-client-web-transports fails with `@parcel/core: Failed to resolve '@pipecat-ai/client-js' from './lib/media-mgmt/mediaManager.ts'` | The workspace's `prepare` scripts try to build every transport, and `small-webrtc-transport` is broken on this branch independently. Add `--ignore-scripts` to the install command (per §2). We only need `moq-transport`'s build, and §3's script does that explicitly. |
| Browser: "MoqTransport requires `relayUrl`" | `/start` didn't return the `moq` block. Bot probably isn't running with `-t moq`. |
| WebTransport handshake fails with `net::ERR_QUIC_PROTOCOL_ERROR` | The fingerprint pinned in the browser doesn't match the cert the bot is presenting. Bot probably restarted (in-process certs are minted fresh each start). Reload the browser tab — `/start` will return the new fingerprint. |
| `pnpm install` errors with corepack signing-key complaint | Install pnpm globally: `npm install -g --force pnpm@10.14.0`. |
| `pnpm install` in voice-ui-kit fails with `ERR_PNPM_FETCH_404 ... @pipecat-ai/moq-transport: Not Found` | The `pnpm.overrides` entry in `voice-ui-kit/package.json` is missing. Run the `npm pkg set "pnpm.overrides[@pipecat-ai/moq-transport]=..."` command from §2 inside `voice-ui-kit/` before retrying `pnpm install`. |
| Bot crashes with `ModuleNotFoundError: No module named 'websockets'` (or similar) at import time | The Python extras in §2 were incomplete. `uv sync --extra moq` alone isn't enough — the example uses Silero VAD, Deepgram STT, Cartesia TTS, and OpenAI LLM; each of those has its own extra. Re-run the full `uv sync` command from §2. |
| Stale voice-ui-kit code after editing | `pnpm -F @pipecat-ai/voice-ui-kit build` was skipped or failed. |

## 7. What the link chain actually does

For when something breaks and the script's output isn't enough.

```
pipecat-prebuilt/client/node_modules/@pipecat-ai/
├── moq-transport     -> ../../../../pipecat-client-web-transports/transports/moq-transport
├── voice-ui-kit      -> ../../../../voice-ui-kit/package
├── client-js         (real copy, npm-installed at 1.8.x — the canonical instance)
└── ...

pipecat-client-web-transports/node_modules/@pipecat-ai/
├── client-js         -> <prebuilt's client-js>  (linked, so moq-transport shares it)
└── ...

voice-ui-kit/package/node_modules/@pipecat-ai/
├── moq-transport     -> ../../../../pipecat-client-web-transports/transports/moq-transport  (pnpm link:)
├── client-js         -> <prebuilt's client-js>  (linked, so voice-ui-kit shares it)
└── ...
```

The whole point is that every layer that imports `@pipecat-ai/client-js`
resolves to the same `dist/` directory. Vite resolves symlinks to real
paths by default (`resolve.preserveSymlinks: false`), so without the
client-js link chain you'd end up with three independent copies and a
broken React context.
