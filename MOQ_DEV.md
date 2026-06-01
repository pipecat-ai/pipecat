# MoQ local dev setup

End-to-end instructions for running the MoQ pipeline locally — a Rust
moq-relay, a pipecat MoQ bot, and the prebuilt React client talking to
both through the locally-built `voice-ui-kit` + `@pipecat-ai/moq-transport`.

The MoQ wire bits are unpublished, so a few `npm link` chains are needed.
Most of the friction is handled by `./scripts/moq-dev-setup.sh` in this
repo.

## 0. Layout

Clone all five repos as siblings under one parent directory. The setup
script assumes this layout:

```
<parent>/
├── pipecat/                          # github.com/pipecat-ai/pipecat
├── pipecat-client-web-transports/    # github.com/pipecat-ai/pipecat-client-web-transports
├── pipecat-prebuilt/                 # internal
├── voice-ui-kit/                     # github.com/pipecat-ai/voice-ui-kit
└── moq-relay/                        # github.com/kixelated/moq-rs
```

The four pipecat repos all live on the **`vp-add-moq-transport`**
branch for this work. `moq-relay` is whichever revision builds —
`main` has worked.

```bash
for repo in pipecat pipecat-client-web-transports pipecat-prebuilt voice-ui-kit; do
  (cd "$repo" && git checkout vp-add-moq-transport)
done
```

## 1. Tooling prerequisites

- Node 18+ and `npm`
- `pnpm` 10+ (the corepack shim on some Node versions has signing-key
  issues — if `pnpm install` fails with a corepack error, install it
  globally instead: `npm install -g --force pnpm@10.14.0`)
- `uv` for the Python side
- `cargo` / Rust toolchain for the relay
- `openssl` (macOS ships with LibreSSL — that works)

## 2. Install all four JS workspaces

Paths assume you start from this `pipecat/` directory and the repos are
siblings:
- `/pipecat`
- `/pipecat-client-web-transports`
- `/voice-ui-kit`
- `/pipecat-prebuilt`

Order doesn't matter; no builds yet. The link chain that ties them
together happens after.

```bash
uv sync --extra moq
( cd ../pipecat-client-web-transports && npm install --ignore-scripts )
( cd ../pipecat-prebuilt/client       && npm install )
( cd ../voice-ui-kit                  && pnpm install )
```

`--ignore-scripts` on the transports repo is intentional: that workspace
runs each transport's `prepare`/build during install, and on this branch
`small-webrtc-transport`'s parcel build fails for an unrelated reason
(imports `@pipecat-ai/client-js` from `lib/` but the workspace root
doesn't declare it as a regular dep). We only need `moq-transport`'s
build, which §4 below does explicitly.

A couple of things to know:

- `voice-ui-kit/package/package.json` declares `@pipecat-ai/moq-transport`
  as a `link:` reference to its sibling repo. `pnpm install` resolves
  that automatically — no extra step. (This will change once
  moq-transport ships to npm.)

## 3. Run the dev setup script

From this `pipecat/` repo root:

```bash
./scripts/moq-dev-setup.sh <path/to/moq-dev/moq>
```

The script does four things:

1. Generates a self-signed cert (`localhost`, 14 days — the WebTransport
   limit) into `pipecat/.moq-certs/`.
2. Symlinks `moq-cert.pem` and `moq-key.pem` into both `pipecat/` and
   the relay dir so each side can find them by relative path.
3. Prints the cert SHA-256 (the bot threads it back through `/start`
   so the browser can pin WebTransport against it).
4. Sets up the `npm link` chain so the prebuilt client picks up the
   local moq-transport + the local voice-ui-kit, and so
   `@pipecat-ai/client-js` is a single physical instance across all
   three packages (prebuilt, voice-ui-kit, moq-transport). Without that
   dedupe, voice-ui-kit's tsc/dts step fails with `Property '_options'
   is protected but type 'Transport' is not a class derived from
   'Transport'` (the linked moq-transport drags in
   pipecat-client-web-transports's hoisted client-js@1.10.x, which
   collides with voice-ui-kit's pnpm-locked 1.8.x), and at runtime the
   React `PipecatClientProvider` context breaks across the duplicates.

The link chain is temporary — delete that block from the script once
both packages ship to npm.

## 4. Build moq-transport + voice-ui-kit

This has to run **after** the link chain is in place — the build needs
the deduped client-js to resolve a single `Transport` type. The setup
script prints these exact commands; for reference:

```bash
( cd ../pipecat-client-web-transports/transports/moq-transport \
    && npm run build )
( cd ../voice-ui-kit \
    && pnpm -F @pipecat-ai/voice-ui-kit build )
```

`pipecat-prebuilt/client` doesn't need an explicit build — Vite's dev
server compiles on demand. If you want a production bundle anyway:
`( cd ../pipecat-prebuilt/client && npm run build )`.

## 5. Run the stack

The setup script prints these at the end; they're listed here for
reference. Three terminals.

### Terminal 1 — moq-relay

```bash
cd ../moq-relay
cargo run --bin moq-relay -- \
  --server-bind '[::]:4080' \
  --tls-cert moq-cert.pem \
  --tls-key moq-key.pem \
  --auth-public ''
```

### Terminal 2 — pipecat bot

From this `pipecat/` repo:

```bash
uv run python examples/transports/transports-moq.py \
  -t moq \
  --moq-cert moq-cert.pem \
  --moq-insecure \
  --moq-path /
```

`--moq-insecure` tells the bot to accept the self-signed cert; the
fingerprint is still pinned on the browser side.

### Terminal 3 — prebuilt client dev server

```bash
cd ../pipecat-prebuilt/client
npm run dev
```

Open <http://localhost:7860>, pick **MoQ** in the transport dropdown.
The `ConsoleTemplate` should drive the full UI — connect/disconnect,
transcript, mic device picker, SessionInfo showing "MoQ".

## 6. Iterating

- **Edited `moq-transport/src/*`?** Rebuild it: `cd
  pipecat-client-web-transports/transports/moq-transport && npm run build`.
  The npm link picks up the new `dist/` immediately; restart the
  prebuilt dev server (Vite caches imports).
- **Edited `voice-ui-kit/package/src/*`?** Rebuild: `pnpm -F
  @pipecat-ai/voice-ui-kit build` (or `build:watch`). Same restart
  caveat for the consumer.
- **Edited `pipecat-prebuilt/client/src/*`?** Vite hot-reloads
  automatically.
- **Edited the pipecat bot?** Restart terminal 2.
- **Cert expired?** Re-run `./scripts/moq-dev-setup.sh ../moq-relay`.
  The script is idempotent.
- **Re-ran `pnpm install` in voice-ui-kit?** pnpm restores its own
  symlink for `@pipecat-ai/client-js`, undoing the dedupe. Re-run the
  setup script before the next `voice-ui-kit` build.

## 7. Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Browser console: `Failed to load transport "moq"` | moq-transport isn't linked into prebuilt. Re-run the setup script and check the "Linking …" output. |
| `Property '_options' is protected` during voice-ui-kit build | Two copies of `@pipecat-ai/client-js` (1.8.x in voice-ui-kit, 1.10.x hoisted in pipecat-client-web-transports). Re-run the setup script before re-trying the build. If you re-ran `pnpm install` in voice-ui-kit after the script, run the script again — pnpm undoes the dedupe symlink. |
| `npm install` in pipecat-client-web-transports fails with `@parcel/core: Failed to resolve '@pipecat-ai/client-js' from './lib/media-mgmt/mediaManager.ts'` | The workspace's `prepare` scripts try to build every transport, and `small-webrtc-transport` is broken on this branch independently. Add `--ignore-scripts` to the install command (per §2). We only need `moq-transport`'s build, and §4 does that explicitly. |
| Browser: "MoqTransport requires `relayUrl`" | `/start` didn't return the `moq` block. Bot probably isn't running with `-t moq`. |
| WebTransport handshake fails with `net::ERR_QUIC_PROTOCOL_ERROR` | Cert expired (14-day limit) or the fingerprint pinned in the browser doesn't match the cert the relay loaded. Re-run the setup script. |
| `pnpm install` errors with corepack signing-key complaint | Install pnpm globally: `npm install -g --force pnpm@10.14.0`. |
| Stale voice-ui-kit code after editing | `pnpm -F @pipecat-ai/voice-ui-kit build` was skipped or failed. |

## 8. What the link chain actually does

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
