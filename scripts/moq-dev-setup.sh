#!/usr/bin/env bash
#
# Generate a self-signed cert for local MOQ dev, symlink it into both the
# relay repo and this pipecat repo, and print the commands to run everything.
#
# Usage:
#   ./scripts/moq-dev-setup.sh /path/to/moq-relay
#

set -euo pipefail

RELAY_DIR="${1:?Usage: $0 /path/to/moq-relay}"
PIPECAT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Resolve relay dir to absolute path
RELAY_DIR="$(cd "$RELAY_DIR" && pwd)"

CERT_DIR="$PIPECAT_DIR/.moq-certs"
CERT_FILE="$CERT_DIR/moq-cert.pem"
KEY_FILE="$CERT_DIR/moq-key.pem"

echo "==> Relay dir:   $RELAY_DIR"
echo "==> Pipecat dir: $PIPECAT_DIR"
echo "==> Cert dir:    $CERT_DIR"
echo

# ---------------------------------------------------------------------------
# 1. Generate cert + key
# ---------------------------------------------------------------------------
mkdir -p "$CERT_DIR"

echo "==> Generating self-signed cert (valid 14 days — MOQ max)..."
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
  -keyout "$KEY_FILE" \
  -out "$CERT_FILE" \
  -days 14 \
  -nodes \
  -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" \
  2>/dev/null

echo "    $CERT_FILE"
echo "    $KEY_FILE"
echo

# ---------------------------------------------------------------------------
# 2. Compute SHA-256 fingerprint (used by WebTransport cert pinning)
# ---------------------------------------------------------------------------
FINGERPRINT=$(openssl x509 -in "$CERT_FILE" -outform der \
  | openssl dgst -sha256 -binary \
  | base64)

echo "==> Certificate SHA-256 fingerprint:"
echo "    $FINGERPRINT"
echo

# ---------------------------------------------------------------------------
# 3. Symlink into relay dir and pipecat dir
# ---------------------------------------------------------------------------
echo "==> Symlinking certs..."

for DIR in "$RELAY_DIR" "$PIPECAT_DIR"; do
  for FILE in "$CERT_FILE" "$KEY_FILE"; do
    BASENAME="$(basename "$FILE")"
    TARGET="$DIR/$BASENAME"
    rm -f "$TARGET"
    ln -s "$FILE" "$TARGET"
    echo "    $TARGET -> $FILE"
  done
done
echo

# ---------------------------------------------------------------------------
# 4. npm-link the unpublished moq-transport into the prebuilt client.
#
# TEMPORARY: remove this block once `@pipecat-ai/moq-transport` is published
# and the prebuilt client depends on it normally.
#
# Two link chains are set up:
#   a) @pipecat-ai/moq-transport (source: pipecat-client-web-transports/transports/moq-transport)
#      -> consumed by pipecat-prebuilt/client
#   b) @pipecat-ai/client-js (source: pipecat-prebuilt/client's installed copy at 1.8.x)
#      -> linked into pipecat-client-web-transports so the linked moq-transport
#         resolves the same Transport class as the consumer (without this,
#         moq-transport picks up client-js@1.10.x from the monorepo and tsc
#         rejects the assignment with "Property '_options' is protected").
#
# Prerequisites: `npm install` has been run in both
#   pipecat-client-web-transports/ and pipecat-prebuilt/client/ already.
# ---------------------------------------------------------------------------
TRANSPORTS_DIR="$PIPECAT_DIR/../pipecat-client-web-transports"
MOQ_PKG_DIR="$TRANSPORTS_DIR/transports/moq-transport"
PREBUILT_CLIENT_DIR="$PIPECAT_DIR/../pipecat-prebuilt/client"
PREBUILT_CLIENT_JS_DIR="$PREBUILT_CLIENT_DIR/node_modules/@pipecat-ai/client-js"

echo "==> Linking moq-transport into pipecat-prebuilt/client (temporary)..."

if [[ ! -d "$MOQ_PKG_DIR" ]]; then
  echo "    skipped: $MOQ_PKG_DIR not found"
elif [[ ! -d "$PREBUILT_CLIENT_DIR/node_modules" ]]; then
  echo "    skipped: run 'npm install' in $PREBUILT_CLIENT_DIR first"
elif [[ ! -d "$PREBUILT_CLIENT_JS_DIR" ]]; then
  echo "    skipped: $PREBUILT_CLIENT_JS_DIR not found (run 'npm install' in $PREBUILT_CLIENT_DIR)"
else
  ( cd "$MOQ_PKG_DIR"          && npm link >/dev/null )
  ( cd "$PREBUILT_CLIENT_DIR"  && npm link @pipecat-ai/moq-transport >/dev/null )
  ( cd "$PREBUILT_CLIENT_JS_DIR" && npm link >/dev/null )
  ( cd "$TRANSPORTS_DIR"       && npm link @pipecat-ai/client-js >/dev/null )
  echo "    @pipecat-ai/moq-transport -> $MOQ_PKG_DIR"
  echo "    @pipecat-ai/client-js     -> $PREBUILT_CLIENT_JS_DIR"
fi
echo

# ---------------------------------------------------------------------------
# 5. Print run commands
# ---------------------------------------------------------------------------
echo "==========================================="
echo " Run these in separate terminals:"
echo "==========================================="
echo
echo "# 1. Start the relay (binds QUIC/WebTransport on UDP [::]:4080)"
echo "cd $RELAY_DIR"
echo "cargo run --bin moq-relay -- \\"
echo "  --server-bind '[::]:4080' \\"
echo "  --tls-cert moq-cert.pem \\"
echo "  --tls-key moq-key.pem \\"
echo "  --auth-public ''"
echo
echo "# 2. Start the bot"
echo "cd $PIPECAT_DIR"
echo "uv run python examples/transports/transports-moq.py \\"
echo "  -t moq \\"
echo "  --moq-cert moq-cert.pem \\"
echo "  --moq-insecure \\"
echo "  --moq-path /"
echo
echo "# 3. Start the prebuilt client dev server"
echo "cd $PIPECAT_DIR/../pipecat-prebuilt/client"
echo "npm run dev"
echo
echo "# 4. Open browser"
echo "open http://localhost:7860"
echo
