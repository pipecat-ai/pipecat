# MoQ local dev setup

End-to-end instructions for running the MoQ pipeline locally — a pipecat
MoQ bot (acting as its own MoQ server) and the prebuilt React client
talking to it through the locally-built `voice-ui-kit` +
`@pipecat-ai/moq-transport`.

The pipecat MoQ transport uses the [`moq-rs`](https://pypi.org/project/moq-rs/)
Python library, so the bot can bind its own UDP socket via `--moq-serve`
and (with `--moq-tls-generate <hostname>`) mint a self-signed cert
in-process. No separate `moq-relay` process, no `openssl` cert dance.

**Why this doc exists.** Neither `@pipecat-ai/moq-transport` nor the
`voice-ui-kit` release that knows about MoQ has shipped to npm yet. The
four repos on disk are the source of truth — the snippet below wires
them together (`npm link` chain + a couple of Vite/pnpm shims) so the
React client resolves the local checkouts instead of the published
versions. Nothing in this doc is meant to survive the packages'
first published release; it should be deleted once both are on npm.

## The short, (well, less short now) short version

Assumes all four repos are cloned as siblings (see §0) and the branch
names are all `vp-add-moq-transport`.

from ~/pipecat:
```bash
export MOQ_BRANCH="${MOQ_BRANCH:-vp-add-moq-transport}"

# 1. Pin every repo to $MOQ_BRANCH.
for repo in . ../pipecat-client-web-transports ../pipecat-prebuilt ../voice-ui-kit; do
  ( cd "$repo" && git fetch origin "$MOQ_BRANCH" && git checkout "$MOQ_BRANCH" )
done

# 2. Install Python + JS deps. voice-ui-kit declares `@pipecat-ai/moq-transport`
#    as a regular dep, but it isn't on npm yet — inject a pnpm override pointing
#    at the local checkout BEFORE `pnpm install`, otherwise the install 404s.
uv sync --extra moq --extra daily --extra silero --extra deepgram \
        --extra cartesia --extra openai --extra runner
( cd ../pipecat-client-web-transports && npm install --ignore-scripts )
( cd ../pipecat-prebuilt/client       && npm install )
( cd ../voice-ui-kit \
    && npm pkg set "pnpm.overrides[@pipecat-ai/moq-transport]=link:../pipecat-client-web-transports/transports/moq-transport" \
    && pnpm install )

# 3. Add the Vite dedupe shim to pipecat-prebuilt/client/vite.config.js.
#    Without this, the linked voice-ui-kit ships its own React + client-js,
#    Vite resolves two copies, and the browser throws "Invalid hook call".
#    Deployed builds don't need this because the npm-published voice-ui-kit
#    resolves everything through the consumer's node_modules.
( cd ../pipecat-prebuilt && git apply <<'PATCH'
diff --git a/client/vite.config.js b/client/vite.config.js
--- a/client/vite.config.js
+++ b/client/vite.config.js
@@ -5,6 +5,18 @@ export default defineConfig({
   base: "./", //Use relative paths so it works at any mount path
   plugins: [react()],
   publicDir: "public",
+  resolve: {
+    // Local-dev only: force every layer through this client's
+    // node_modules so the linked voice-ui-kit doesn't drag in a
+    // second React or a second @pipecat-ai/client-js.
+    dedupe: [
+      "react",
+      "react-dom",
+      "react/jsx-runtime",
+      "@pipecat-ai/client-js",
+      "@pipecat-ai/client-react",
+    ],
+  },
   server: {
     allowedHosts: true, // Allows external connections like ngrok
     proxy: {
PATCH
)

# 4. Link chain: point pipecat-prebuilt at the local moq-transport +
#    voice-ui-kit, and dedupe `@pipecat-ai/client-js` to a single physical
#    copy so tsc doesn't see two `Transport` types during voice-ui-kit's
#    build.
TRANSPORTS_DIR="$(pwd)/../pipecat-client-web-transports"
MOQ_PKG_DIR="$TRANSPORTS_DIR/transports/moq-transport"
PREBUILT_CLIENT_DIR="$(pwd)/../pipecat-prebuilt/client"
PREBUILT_CLIENT_JS_DIR="$PREBUILT_CLIENT_DIR/node_modules/@pipecat-ai/client-js"
VUK_PKG_DIR="$(pwd)/../voice-ui-kit/package"
VUK_CLIENT_JS_LINK="$VUK_PKG_DIR/node_modules/@pipecat-ai/client-js"

# Register the unpublished packages + the canonical client-js globally.
# --ignore-scripts avoids re-running each package's build/prepare here.
( cd "$MOQ_PKG_DIR"           && npm link --ignore-scripts )
( cd "$VUK_PKG_DIR"            && npm link --ignore-scripts )
( cd "$PREBUILT_CLIENT_JS_DIR" && npm link --ignore-scripts )

# Pull them into the prebuilt client and dedupe client-js into the
# transports workspace (npm-managed, so `npm link` cooperates).
# --ignore-scripts on the transports repo is critical: it's a workspace
# root and a vanilla `npm link` would trigger every workspace's `prepare`
# lifecycle and try to rebuild all the other transports.
( cd "$PREBUILT_CLIENT_DIR"    && npm link --ignore-scripts @pipecat-ai/moq-transport @pipecat-ai/voice-ui-kit )
( cd "$TRANSPORTS_DIR"         && npm link --ignore-scripts @pipecat-ai/client-js )

# voice-ui-kit is a pnpm workspace; `npm link` errors against its
# managed node_modules. Replace the client-js symlink manually instead.
mkdir -p "$(dirname "$VUK_CLIENT_JS_LINK")"
rm -rf "$VUK_CLIENT_JS_LINK"
ln -s "$PREBUILT_CLIENT_JS_DIR" "$VUK_CLIENT_JS_LINK"

# `npm install` inside transports/moq-transport drops a non-hoisted
# client-js into moq-transport/node_modules. When tsc resolves
# MoqTransport's `Transport` import while building voice-ui-kit — which
# is a consumer of moq-transport via pnpm `link:` — it walks node_modules
# from moq-transport's real location, finds this local copy FIRST, and
# ends up resolving to a different physical path than voice-ui-kit's own
# client-js. Result: two `Transport` types, protected `_options`
# collision, dts build fails. Delete the local copy so resolution walks
# up to the deduped root.
rm -rf "$MOQ_PKG_DIR/node_modules/@pipecat-ai/client-js"

# 5. Build the linked packages so their dist/ reflects local source.
( cd "$MOQ_PKG_DIR"     && npm run build )
( cd ../voice-ui-kit    && pnpm -F @pipecat-ai/voice-ui-kit build )
```

You should now be able to start the bot and the prebuilt client — see §3.

## 0. Layout

Clone all four repos as siblings under one parent directory:

```
<parent>/
├── pipecat/                          # github.com/pipecat-ai/pipecat
├── pipecat-client-web-transports/    # github.com/pipecat-ai/pipecat-client-web-transports
├── pipecat-prebuilt/                 # github.com/pipecat-ai/pipecat-prebuilt
└── voice-ui-kit/                     # github.com/pipecat-ai/voice-ui-kit
```

All four repos live on the same MoQ development branch — currently
`vp-add-moq-transport`. Export `MOQ_BRANCH` to override.

## 1. Tooling prerequisites

- Node 18+ and `npm`
- `pnpm` 10+ (the corepack shim on some Node versions has signing-key
  issues — if `pnpm install` fails with a corepack error, install it
  globally instead: `npm install -g --force pnpm@10.14.0`)
- `uv` for the Python side

## 2. What the snippet is doing

The block in "The short, short version" is meant to be pasted into a
terminal sitting in the pipecat repo root. Here's what each step does:

**Step 1 — check out the MoQ branch.** All four repos need to be on the
same branch (`$MOQ_BRANCH`, default `vp-add-moq-transport`). Re-run this
any time you suspect the branches have drifted.

**Step 2 — install deps.** The Python extras cover the pipeline in
`examples/transports/transports-moq.py`: `silero` for VAD, `deepgram`
for STT, `cartesia` for TTS, `openai` for the LLM, `runner` for the
FastAPI server that hosts `/start` + the prebuilt UI, `moq` for the
transport itself (pulls in `moq-rs` + `cryptography`), and `daily` for
the example's unconditional `DailyParams` import at the top.

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
build, which step 5 handles directly.

The `npm pkg set` line injects a `pnpm.overrides` entry in
`voice-ui-kit/package.json` pointing `@pipecat-ai/moq-transport` at the
sibling checkout, so `pnpm install` resolves it locally instead of
404-ing on the registry. Drop the override once moq-transport ships to
npm.

**Step 3 — Vite dedupe shim.** The `resolve.dedupe` block forces every
layer through the prebuilt client's `node_modules`. Without it the
linked voice-ui-kit's own React (pnpm-locked 19.2.1) and prebuilt's
React (19.2.x from npm) both load and hooks break across the boundary
("Invalid hook call" at runtime). This is a link-chain artifact — a
deployed npm build resolves everything through the consumer, so the
production `vite.config.js` doesn't (and shouldn't) carry this block.

**Step 4 — npm link chain.** Wires the unpublished packages into the
prebuilt client and ensures every layer's `@pipecat-ai/client-js`
import resolves to the same physical directory. Without that dedupe:

- voice-ui-kit's tsc/dts step fails with `Property '_options' is
  protected but type 'Transport' is not a class derived from 'Transport'`
  (linked moq-transport drags in the transports repo's hoisted
  client-js@1.10.x, which collides with voice-ui-kit's pnpm-locked
  1.8.x).
- At runtime, the React `PipecatClientProvider` context breaks across
  the duplicates.

**Step 5 — build.** Now that the link chain resolves a single
`Transport` type, the dts build passes. `pipecat-prebuilt/client`
doesn't need an explicit build; Vite compiles on demand.

## 3. Run the stack

Two terminals.

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

Server mode refuses to start without explicit TLS config — pass either
`--moq-tls-generate <hostname>` (dev) or `--moq-tls-cert /path/to/cert.pem
--moq-tls-key /path/to/key.pem` (prod). This is intentional, to avoid
silently shipping a self-signed cert into a production deploy.

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

## 4. Iterating

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
  step 4 of the setup snippet before the next `voice-ui-kit` build.

## 5. Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Browser console: `Failed to load transport "moq"` | moq-transport isn't linked into prebuilt. Re-run step 4 of the setup snippet. |
| No **MoQ** entry in the transport dropdown | You're at `http://localhost:7860/client/`, which is the published `pipecat-ai-prebuilt` wheel served by the pipecat runner — it predates our MoQ work. Open the Vite dev server URL instead (usually `http://localhost:5173`); that's what `npm run dev` serves with your local edits. |
| Vite URL is a blank page, console says `Invalid hook call` / `Cannot read properties of null (reading 'useEffect')` | Two React instances. The linked voice-ui-kit ships its own pnpm-locked React and prebuilt has its own; hooks break across the boundary. Fix is the Vite `resolve.dedupe` block from step 3 of the setup snippet — re-run that step. |
| `Property '_options' is protected` during voice-ui-kit build | Two copies of `@pipecat-ai/client-js` (1.8.x in voice-ui-kit, 1.10.x hoisted in pipecat-client-web-transports). Re-run steps 4–5 of the setup snippet. If you re-ran `pnpm install` in voice-ui-kit after the setup, run step 4 again — pnpm undoes the dedupe symlink. |
| `npm install` in pipecat-client-web-transports fails with `@parcel/core: Failed to resolve '@pipecat-ai/client-js' from './lib/media-mgmt/mediaManager.ts'` | The workspace's `prepare` scripts try to build every transport, and `small-webrtc-transport` is broken on this branch independently. Add `--ignore-scripts` to the install command (per the snippet). We only need `moq-transport`'s build, and step 5 does that explicitly. |
| `git apply` in step 3 fails with `patch does not apply` | The vite.config.js in `pipecat-prebuilt/client` has drifted from what the patch expects. Either update `$MOQ_BRANCH` if it lags behind, or apply the `resolve.dedupe` block by hand. |
| Browser: "MoqTransport requires `relayUrl`" | `/start` didn't return the `moq` block. Bot probably isn't running with `-t moq`. |
| WebTransport handshake fails with `net::ERR_QUIC_PROTOCOL_ERROR` | The fingerprint pinned in the browser doesn't match the cert the bot is presenting. Bot probably restarted (in-process certs are minted fresh each start). Reload the browser tab — `/start` will return the new fingerprint. |
| `pnpm install` errors with corepack signing-key complaint | Install pnpm globally: `npm install -g --force pnpm@10.14.0`. |
| `pnpm install` in voice-ui-kit fails with `ERR_PNPM_FETCH_404 ... @pipecat-ai/moq-transport: Not Found` | The `pnpm.overrides` entry in `voice-ui-kit/package.json` is missing. Run the `npm pkg set` command from step 2 inside `voice-ui-kit/` before retrying `pnpm install`. |
| Bot crashes with `ModuleNotFoundError: No module named 'websockets'` (or similar) at import time | The Python extras in step 2 were incomplete. `uv sync --extra moq` alone isn't enough — the example uses Silero VAD, Deepgram STT, Cartesia TTS, and OpenAI LLM; each has its own extra. Re-run the full `uv sync` command. |
| Stale voice-ui-kit code after editing | `pnpm -F @pipecat-ai/voice-ui-kit build` was skipped or failed. |
| Safari console: `WebSocket connection to 'wss://localhost:4080/moq' failed` and no audio | Use Chrome/Chromium for local `--moq-serve` dev. Two compounding issues: (a) Safari <18 has no WebTransport at all; (b) Safari 18+ has WebTransport but doesn't support `serverCertificateHashes`, so the in-process self-signed cert can't be pinned. `@moq/net` then falls back to WebSocket, but `--moq-serve` only binds a UDP/QUIC socket — there's no `wss://` listener on :4080 — so the fallback fails too. Safari works against an external relay with a CA-signed cert. |

## 6. What the link chain actually does

For when something breaks and the setup snippet's output isn't enough.

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

## 7. Unwinding for a merge / publish

None of the shims introduced here should be committed to their respective
repos. To confirm a clean state on each branch:

- `pipecat/`: no `MOQ_DEV.md`, no `scripts/moq-dev-setup.sh`.
- `pipecat-prebuilt/client/vite.config.js`: no `resolve.dedupe` block.
- `voice-ui-kit/package.json`: no `pnpm.overrides["@pipecat-ai/moq-transport"]`
  entry (the `npm pkg set` in step 2 mutates this file — revert with
  `npm pkg delete "pnpm.overrides[@pipecat-ai/moq-transport]"` before
  committing).
- No stray `node_modules/@pipecat-ai/*` symlinks — `npm unlink` in each
  repo, or just wipe `node_modules/` and reinstall.

Once `@pipecat-ai/moq-transport` and the next `voice-ui-kit` release
(with MoQ support) both ship to npm, this whole document can be deleted.
