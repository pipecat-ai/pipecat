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
# 4. Print run commands
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
echo "# 3. Open browser"
echo "open http://localhost:7860"
echo
