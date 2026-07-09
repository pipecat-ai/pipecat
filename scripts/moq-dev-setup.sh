#!/usr/bin/env bash
#
# One-shot local dev setup for the MoQ pipeline. Wires the four repos
# together so the React prebuilt client resolves the local checkouts of
# `@pipecat-ai/moq-transport` and the MoQ-enabled `voice-ui-kit` instead
# of the (currently non-existent) npm-published versions.
#
# See MOQ_DEV.md for the full context. TL;DR of what this does, in order:
#
#   1. `git fetch` + `git checkout $MOQ_BRANCH` in all four sibling repos.
#   2. `uv sync` the pipecat extras; `npm install` / `pnpm install` in
#      each JS repo. Injects a `pnpm.overrides` entry in voice-ui-kit
#      pointing @pipecat-ai/moq-transport at the local checkout so
#      `pnpm install` doesn't 404.
#   3. Applies a Vite dedupe shim to pipecat-prebuilt/client/vite.config.js
#      (skipped if already applied) so the linked voice-ui-kit doesn't
#      pull in a second React / client-js copy.
#   4. `npm link` chain across the three JS repos; ensures every layer's
#      `@pipecat-ai/client-js` import resolves to the same physical
#      directory (otherwise voice-ui-kit's dts build fails with a
#      protected `_options` collision).
#   5. Builds `moq-transport` and `voice-ui-kit` so their `dist/`
#      reflects local source.
#
# Env vars:
#   MOQ_BRANCH   (default "vp-add-moq-transport") — branch to check out in
#                every sibling repo.
#   SKIP_CHECKOUT=1 — skip step 1 (useful when iterating without touching
#                    branches).
#
# TEMPORARY: delete this script (and MOQ_DEV.md) once @pipecat-ai/moq-transport
# ships and voice-ui-kit's next release (with MoQ support) is on npm.

set -euo pipefail

MOQ_BRANCH="${MOQ_BRANCH:-vp-add-moq-transport}"
PIPECAT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PARENT_DIR="$(cd "$PIPECAT_DIR/.." && pwd)"

TRANSPORTS_DIR="$PARENT_DIR/pipecat-client-web-transports"
MOQ_PKG_DIR="$TRANSPORTS_DIR/transports/moq-transport"
PREBUILT_DIR="$PARENT_DIR/pipecat-prebuilt"
PREBUILT_CLIENT_DIR="$PREBUILT_DIR/client"
PREBUILT_CLIENT_JS_DIR="$PREBUILT_CLIENT_DIR/node_modules/@pipecat-ai/client-js"
VUK_DIR="$PARENT_DIR/voice-ui-kit"
VUK_PKG_DIR="$VUK_DIR/package"
VUK_CLIENT_JS_LINK="$VUK_PKG_DIR/node_modules/@pipecat-ai/client-js"

echo "==> pipecat:            $PIPECAT_DIR"
echo "==> transports:         $TRANSPORTS_DIR"
echo "==> pipecat-prebuilt:   $PREBUILT_DIR"
echo "==> voice-ui-kit:       $VUK_DIR"
echo "==> MOQ_BRANCH:         $MOQ_BRANCH"
echo

# Preflight: all four repos must be siblings under $PARENT_DIR.
for d in "$PIPECAT_DIR" "$TRANSPORTS_DIR" "$PREBUILT_DIR" "$VUK_DIR"; do
  if [[ ! -d "$d/.git" ]]; then
    echo "ERROR: expected a git repo at $d" >&2
    echo "       clone all four repos as siblings — see MOQ_DEV.md §0." >&2
    exit 1
  fi
done

# `--cleanup`: stash+drop the working-tree shims this script leaves behind
# in each repo (vite dedupe, pnpm override, lock file churn). Targets
# specific paths so unrelated in-flight work isn't touched. Compares the
# stash count before/after so we only drop the stash we just created.
if [[ "${1:-}" == "--cleanup" ]]; then
  _cleanup_repo() {
    local repo="$1"; shift
    local before after
    before=$(git -C "$repo" stash list | wc -l)
    git -C "$repo" stash push --include-untracked -m "moq-dev-cleanup" -- "$@" >/dev/null 2>&1 || true
    after=$(git -C "$repo" stash list | wc -l)
    if (( after > before )); then
      git -C "$repo" stash drop stash@{0} >/dev/null
      echo "    dropped in $(basename "$repo"): $*"
    else
      echo "    nothing to drop in $(basename "$repo")"
    fi
  }
  echo "==> Cleaning up working-tree shims..."
  _cleanup_repo "$PREBUILT_DIR"   client/vite.config.js client/package.json client/package-lock.json
  _cleanup_repo "$TRANSPORTS_DIR" package-lock.json
  _cleanup_repo "$VUK_DIR"        package.json pnpm-lock.yaml
  exit 0
fi

# ---------------------------------------------------------------------------
# 1. Pin every repo to $MOQ_BRANCH.
# ---------------------------------------------------------------------------
if [[ "${SKIP_CHECKOUT:-0}" != "1" ]]; then
  echo "==> [1/5] Checking out $MOQ_BRANCH in every repo..."
  for d in "$PIPECAT_DIR" "$TRANSPORTS_DIR" "$PREBUILT_DIR" "$VUK_DIR"; do
    ( cd "$d" && git fetch origin "$MOQ_BRANCH" && git checkout "$MOQ_BRANCH" )
  done
  echo
else
  echo "==> [1/5] Skipping branch checkout (SKIP_CHECKOUT=1)."
  echo
fi

# ---------------------------------------------------------------------------
# 2. Install deps.
#
# The pnpm override on voice-ui-kit is required because
# @pipecat-ai/moq-transport is not on npm yet — without the override,
# `pnpm install` 404s trying to resolve it.
#
# --ignore-scripts on the transports repo dodges an unrelated
# small-webrtc-transport parcel build failure. We only need
# moq-transport's build, and step 5 does that explicitly.
#
# Note on the prebuilt client-js pin: prebuilt's committed package.json
# declares `^1.11.0`. We pin to 1.12.0 explicitly so the whole tree
# canonicalizes there. Why 1.12.0 specifically:
#   - Declares RTVI protocol `2.0.0` in client-ready (older 1.x versions
#     declare `1.4.0` and force the server observer onto its legacy v1
#     bot-output path — which the runtime v2 aggregator in
#     client-react ≥1.5 then mis-parses as duplicated sentences).
#   - Exports `findElementByRef` / `setAboutClient` (missing in 1.8.x),
#     which client-react's compiled dist imports statically.
#   - Satisfies every downstream peer we care about (moq-transport's
#     ~1.10 warns but works fine — Transport surface hasn't changed).
# `npm pkg set` mutates the declared range in place; this survives
# step 4's `npm link` chain (unlike `npm install --no-save`, which
# npm's dep-tree reconciliation silently unwinds). --cleanup stashes
# prebuilt/client/package.json so the pin doesn't get committed.
# ---------------------------------------------------------------------------
echo "==> [2/5] Installing Python + JS deps..."
( cd "$PIPECAT_DIR" && uv sync \
    --extra moq --extra daily --extra silero --extra deepgram \
    --extra cartesia --extra openai --extra runner )
( cd "$TRANSPORTS_DIR"    && npm install --ignore-scripts )
# Prebuilt: pin the client-js range to 1.12.0 (see the design note above
# step 2 for why). `npm pkg set` mutates the declared range in place;
# this survives step 4's `npm link` chain (unlike `npm install --no-save`,
# which npm's dep-tree reconciliation silently unwinds).
# --legacy-peer-deps bypasses peer-dep warnings from the sibling
# transports whose peer ranges don't perfectly overlap.
( cd "$PREBUILT_CLIENT_DIR" \
    && npm pkg set "dependencies[@pipecat-ai/client-js]=1.12.0" \
    && npm install --legacy-peer-deps )
# voice-ui-kit's pnpm workspace: `pnpm.overrides[@pipecat-ai/moq-transport]`
# points at the local moq-transport checkout (unpublished — the install
# would 404 without it). `pnpm.overrides[@pipecat-ai/client-js]` points
# at prebuilt's node_modules copy — this is what makes every layer of
# voice-ui-kit's tsc/dts build resolve to the SAME physical client-js
# file as prebuilt (and by extension moq-transport, via step 4's link
# chain into the transports repo). Without it, voice-ui-kit's direct
# imports, client-react's peer resolution, and moq-transport's Transport
# all land on different physical paths and tsc rejects the resulting
# type mismatches with the "protected `_options`" collision.
( cd "$VUK_DIR" \
    && npm pkg set "pnpm.overrides[@pipecat-ai/moq-transport]=link:../pipecat-client-web-transports/transports/moq-transport" \
    && npm pkg set "pnpm.overrides[@pipecat-ai/client-js]=link:../pipecat-prebuilt/client/node_modules/@pipecat-ai/client-js" \
    && pnpm install )
echo

# ---------------------------------------------------------------------------
# 3. Vite dedupe shim.
#
# Without this, the linked voice-ui-kit ships its own React + client-js,
# Vite resolves two copies, and the browser throws "Invalid hook call".
# Deployed builds don't need this because the npm-published voice-ui-kit
# resolves everything through the consumer's node_modules — so the shim
# lives here rather than being committed to vite.config.js.
# ---------------------------------------------------------------------------
echo "==> [3/5] Adding Vite dedupe shim to pipecat-prebuilt/client/vite.config.js..."
if grep -q '"@pipecat-ai/client-react"' "$PREBUILT_CLIENT_DIR/vite.config.js" 2>/dev/null; then
  echo "    already applied — skipping."
else
  ( cd "$PREBUILT_DIR" && git apply <<'PATCH'
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
  echo "    applied."
fi
echo

# ---------------------------------------------------------------------------
# 4. Link chain — point pipecat-prebuilt at the local moq-transport +
#    voice-ui-kit, and dedupe @pipecat-ai/client-js to a single physical
#    copy across all three JS repos.
#
# Without the client-js dedupe:
#   * voice-ui-kit's tsc/dts step fails with
#     `Property '_options' is protected but type 'Transport' is not a
#      class derived from 'Transport'` (linked moq-transport drags in
#     the transports repo's hoisted client-js, which collides with
#     voice-ui-kit's pnpm-locked copy).
#   * At runtime, the React `PipecatClientProvider` context breaks
#     across the duplicates.
# ---------------------------------------------------------------------------
echo "==> [4/5] Wiring npm link chain..."

# Register the unpublished packages + the canonical client-js globally.
# --ignore-scripts avoids re-running each package's build/prepare here.
( cd "$MOQ_PKG_DIR"           && npm link --ignore-scripts >/dev/null )
( cd "$VUK_PKG_DIR"            && npm link --ignore-scripts >/dev/null )
( cd "$PREBUILT_CLIENT_JS_DIR" && npm link --ignore-scripts >/dev/null )

# Pull them into the prebuilt client and dedupe client-js into the
# transports workspace (npm-managed, so `npm link` cooperates).
# --ignore-scripts on the transports repo is critical: it's a workspace
# root and a vanilla `npm link` would trigger every workspace's `prepare`
# lifecycle and try to rebuild all the other transports.
( cd "$PREBUILT_CLIENT_DIR" && npm link --ignore-scripts --legacy-peer-deps \
    @pipecat-ai/moq-transport @pipecat-ai/voice-ui-kit >/dev/null )

# Transports repo hoists its own client-js (workspace transports declare
# it at ^1.12.0). `npm link @pipecat-ai/client-js` here becomes a no-op
# because the hoisted 1.12.0 already satisfies the peer, and moq-transport
# would then resolve `Transport` from a different physical path than
# voice-ui-kit — dts collision. Force the symlink manually.
rm -rf "$TRANSPORTS_DIR/node_modules/@pipecat-ai/client-js"
ln -s "$PREBUILT_CLIENT_JS_DIR" "$TRANSPORTS_DIR/node_modules/@pipecat-ai/client-js"

# NOTE: we deliberately do NOT force voice-ui-kit's client-js symlink
# onto prebuilt's copy. voice-ui-kit's pnpm workspace has client-js
# 1.8.x from the lockfile, prebuilt's canonical is 1.10.0. Forcing the
# symlink here makes voice-ui-kit's direct imports and client-react's
# peer-resolved copy diverge (different physical paths), and tsc's dts
# build fails with the protected `_options` collision. Letting pnpm's
# natural resolution stand keeps voice-ui-kit's build internally
# consistent. Vite's `resolve.dedupe` (step 3) unifies everyone onto
# prebuilt's copy at bundle time, which is what actually matters for
# the browser.

# `npm install` inside transports/moq-transport can drop a non-hoisted
# client-js into moq-transport/node_modules. When tsc resolves
# MoqTransport's `Transport` import while building voice-ui-kit — which
# is a consumer of moq-transport via pnpm `link:` — it walks node_modules
# from moq-transport's real location, finds this local copy FIRST, and
# ends up resolving to a different physical path than voice-ui-kit's own
# client-js. Result: two `Transport` types, protected `_options` collision,
# dts build fails. Delete the local copy so resolution walks up to the
# deduped root.
rm -rf "$MOQ_PKG_DIR/node_modules/@pipecat-ai/client-js"

echo "    @pipecat-ai/moq-transport -> $MOQ_PKG_DIR"
echo "    @pipecat-ai/voice-ui-kit  -> $VUK_PKG_DIR"
echo "    @pipecat-ai/client-js     -> $PREBUILT_CLIENT_JS_DIR (pinned to 1.8.0, shared)"
echo

# ---------------------------------------------------------------------------
# 5. Build the linked packages so their dist/ reflects local source.
#    (pipecat-prebuilt/client doesn't need an explicit build; Vite
#    compiles on demand.)
# ---------------------------------------------------------------------------
echo "==> [5/5] Building linked packages..."
( cd "$MOQ_PKG_DIR" && npm run build >/dev/null )
echo "    @pipecat-ai/moq-transport dist/ up to date."
( cd "$VUK_DIR" && pnpm -F @pipecat-ai/voice-ui-kit build >/dev/null )
echo "    @pipecat-ai/voice-ui-kit dist/ up to date."
echo

# ---------------------------------------------------------------------------
# Next-step hint
# ---------------------------------------------------------------------------
cat <<EOF
==========================================
 Done. Two terminals to run the stack:
==========================================

# 1. Bot (self-serves as the MoQ server)
cd $PIPECAT_DIR
uv run python examples/transports/transports-moq.py \\
    -t moq --moq-serve --moq-tls-generate localhost

# 2. Prebuilt client dev server (Vite, usually port 5173)
cd $PREBUILT_CLIENT_DIR
npm run dev

Open the Vite dev URL (http://localhost:5173/ NOT http://localhost:7860/client/, 
which serves the published pipecat-ai-prebuilt wheel without MoQ). Pick "MoQ" in the
transport dropdown.
EOF
