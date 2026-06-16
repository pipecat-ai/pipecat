#!/usr/bin/env bash
#
# Wire the unpublished MoQ-related packages into pipecat-prebuilt/client
# for local dev, then build them.
#
# The pipecat MoQ transport now self-serves via `--moq-serve` + a TLS
# arg (`--moq-tls-generate <hostname>` for a dev self-signed cert, or
# `--moq-tls-cert/--moq-tls-key` for a CA-signed one). The bot binds its
# own UDP socket and (in dev) mints the cert in-process, so you no
# longer need a separate moq-relay process or an `openssl` cert dance.
# This script's only job is the link chain that lets the React
# prebuilt client pick up the local source of:
#
#   * @pipecat-ai/moq-transport (unpublished)
#   * @pipecat-ai/voice-ui-kit  (local source ahead of the next release)
#   * @pipecat-ai/client-js     (deduped to a single physical instance)
#
# Usage:
#   ./scripts/moq-dev-setup.sh
#
# TEMPORARY: delete this script once @pipecat-ai/moq-transport ships and
# voice-ui-kit's next release (with MoQ support) is on npm.
#

set -euo pipefail

PIPECAT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Pipecat dir: $PIPECAT_DIR"
echo

# ---------------------------------------------------------------------------
# 1. Link the unpublished moq-transport + the local voice-ui-kit into the
#    prebuilt client, and dedupe @pipecat-ai/client-js across all three.
#
# This MUST run BEFORE building voice-ui-kit, so its tsc/dts step resolves
# a single @pipecat-ai/client-js. Otherwise it picks up its own (1.8.x via
# the pnpm lockfile) AND pipecat-client-web-transports' hoisted copy
# (currently 1.10.x) and fails with:
#   "Property '_options' is protected but type 'Transport' is not a class
#    derived from 'Transport'"
#
# Link chains set up:
#   a) @pipecat-ai/moq-transport (source: pipecat-client-web-transports/transports/moq-transport)
#      -> consumed by pipecat-prebuilt/client
#   b) @pipecat-ai/voice-ui-kit  (source: voice-ui-kit/package)
#      -> consumed by pipecat-prebuilt/client (replaces the npm-installed
#         release with the local build that knows about transportType="moq").
#   c) @pipecat-ai/client-js     (canonical: pipecat-prebuilt/client's
#      installed copy at 1.8.x) -> linked into pipecat-client-web-transports
#      and voice-ui-kit/package so every layer in the build resolves the
#      same Transport / PipecatClient class instance.
#
# voice-ui-kit/package is a pnpm workspace; `npm link` errors against its
# managed node_modules. We replace the client-js symlink with a manual
# `ln -s` instead, which produces an equivalent symlink on disk.
#
# Prerequisites: `npm install` in pipecat-client-web-transports/ and
# pipecat-prebuilt/client/, and `pnpm install` in voice-ui-kit/.
# ---------------------------------------------------------------------------
TRANSPORTS_DIR="$PIPECAT_DIR/../pipecat-client-web-transports"
MOQ_PKG_DIR="$TRANSPORTS_DIR/transports/moq-transport"
PREBUILT_CLIENT_DIR="$PIPECAT_DIR/../pipecat-prebuilt/client"
PREBUILT_CLIENT_JS_DIR="$PREBUILT_CLIENT_DIR/node_modules/@pipecat-ai/client-js"
VUK_PKG_DIR="$PIPECAT_DIR/../voice-ui-kit/package"
VUK_CLIENT_JS_LINK="$VUK_PKG_DIR/node_modules/@pipecat-ai/client-js"

echo "==> Linking moq-transport + voice-ui-kit + deduping client-js (temporary)..."

if [[ ! -d "$MOQ_PKG_DIR" ]]; then
  echo "    skipped: $MOQ_PKG_DIR not found"
elif [[ ! -d "$PREBUILT_CLIENT_DIR/node_modules" ]]; then
  echo "    skipped: run 'npm install' in $PREBUILT_CLIENT_DIR first"
elif [[ ! -d "$PREBUILT_CLIENT_JS_DIR" ]]; then
  echo "    skipped: $PREBUILT_CLIENT_JS_DIR not found (run 'npm install' in $PREBUILT_CLIENT_DIR)"
elif [[ ! -d "$VUK_PKG_DIR/node_modules" ]]; then
  echo "    skipped: $VUK_PKG_DIR/node_modules not found (run 'pnpm install' in $(dirname "$VUK_PKG_DIR"))"
else
  # Register the unpublished packages globally. --ignore-scripts so npm
  # doesn't try to re-run each package's build/prepare here.
  ( cd "$MOQ_PKG_DIR"            && npm link --ignore-scripts >/dev/null )
  ( cd "$VUK_PKG_DIR"             && npm link --ignore-scripts >/dev/null )
  ( cd "$PREBUILT_CLIENT_JS_DIR"  && npm link --ignore-scripts >/dev/null )

  # Pull them into the prebuilt client.
  ( cd "$PREBUILT_CLIENT_DIR"     && npm link --ignore-scripts @pipecat-ai/moq-transport @pipecat-ai/voice-ui-kit >/dev/null )

  # Dedupe client-js into the transports repo (npm-managed, npm link works).
  # --ignore-scripts is critical here: pipecat-client-web-transports is a
  # workspace root, and a vanilla `npm link` would trigger every workspace's
  # prepare lifecycle and try to rebuild all the other transports.
  ( cd "$TRANSPORTS_DIR"          && npm link --ignore-scripts @pipecat-ai/client-js >/dev/null )

  # Dedupe client-js into voice-ui-kit (pnpm-managed — replace its symlink
  # manually since `npm link` doesn't cooperate with pnpm's node_modules).
  mkdir -p "$(dirname "$VUK_CLIENT_JS_LINK")"
  rm -rf "$VUK_CLIENT_JS_LINK"
  ln -s "$PREBUILT_CLIENT_JS_DIR" "$VUK_CLIENT_JS_LINK"

  # `npm install` inside transports/moq-transport (run during the c5
  # rewrite, and again any time someone refreshes deps there) drops a
  # non-hoisted client-js into moq-transport/node_modules. When tsc/dts
  # resolves MoqTransport's `Transport` import while building voice-ui-kit
  # — which is a *consumer* of moq-transport via pnpm `link:` — it walks
  # node_modules from moq-transport's REAL location, finds this local
  # copy FIRST, and ends up resolving to a different physical path than
  # voice-ui-kit's own client-js. Result: two `Transport` types, the
  # protected `_options` collision, and the dts build fails. Delete the
  # local copy so resolution walks up to the deduped root.
  MOQ_PKG_LOCAL_CLIENT_JS="$MOQ_PKG_DIR/node_modules/@pipecat-ai/client-js"
  if [[ -d "$MOQ_PKG_LOCAL_CLIENT_JS" ]]; then
    rm -rf "$MOQ_PKG_LOCAL_CLIENT_JS"
  fi

  echo "    @pipecat-ai/moq-transport -> $MOQ_PKG_DIR"
  echo "    @pipecat-ai/voice-ui-kit  -> $VUK_PKG_DIR"
  echo "    @pipecat-ai/client-js     -> $PREBUILT_CLIENT_JS_DIR (shared into transports + voice-ui-kit)"
fi
echo

# ---------------------------------------------------------------------------
# 2. Build the linked packages so their dist/ reflects local source.
#
# The link chain is in place by this point, so tsc resolves a single
# Transport type and the dts build passes.
# ---------------------------------------------------------------------------
if [[ -d "$MOQ_PKG_DIR" ]]; then
  echo "==> Building @pipecat-ai/moq-transport..."
  ( cd "$MOQ_PKG_DIR" && npm run build >/dev/null )
  echo "    OK ($MOQ_PKG_DIR/dist)"
  echo
fi

if [[ -d "$VUK_PKG_DIR" ]]; then
  echo "==> Building @pipecat-ai/voice-ui-kit..."
  ( cd "$(dirname "$VUK_PKG_DIR")" && pnpm -F @pipecat-ai/voice-ui-kit build >/dev/null )
  echo "    OK ($VUK_PKG_DIR/dist)"
  echo
fi

# ---------------------------------------------------------------------------
# 3. Print run commands
# ---------------------------------------------------------------------------
echo "==========================================="
echo " Next: run these in separate terminals"
echo "==========================================="
echo
echo "# 1. Start the bot (self-serves as the MOQ server: --moq-serve binds"
echo "#    the UDP socket; --moq-tls-generate localhost mints a self-signed"
echo "#    cert in-process for that hostname and threads the fingerprint to"
echo "#    the browser via /start. No separate moq-relay process needed.)"
echo "cd $PIPECAT_DIR"
echo "uv run python examples/transports/transports-moq.py \\"
echo "    -t moq --moq-serve --moq-tls-generate localhost"
echo
echo "# 2. Start the prebuilt client dev server (Vite — usually port 5173)"
echo "cd $PIPECAT_DIR/../pipecat-prebuilt/client"
echo "npm run dev"
echo
echo "# 3. Open the Vite dev URL printed in terminal 2 (NOT the bot's"
echo "#    http://localhost:7860/client/, which serves the published"
echo "#    pipecat-ai-prebuilt wheel without MoQ)."
echo "open http://localhost:5173"
echo
