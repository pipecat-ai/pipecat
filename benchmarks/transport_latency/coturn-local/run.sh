#!/usr/bin/env bash
#
# Run coturn in the FOREGROUND (logs stream here; ctrl-c stops it), for the
# Pipecat transport-latency benchmark's "relayed" WebRTC scenario. Mirrors
# `moq-relay-dev.sh relay` (the local MoQ relay helper).
#
# Env vars:
#   COTURN_IMAGE=coturn/coturn:4.7   image tag (pin for reproducibility)
#   TURN_PORT=3478                   STUN/TURN listening port on the host

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COTURN_IMAGE="${COTURN_IMAGE:-coturn/coturn:4.7}"
TURN_PORT="${TURN_PORT:-3478}"
CONTAINER_NAME="pipecat-coturn-dev"

command -v docker >/dev/null || { echo "ERROR: docker is required" >&2; exit 1; }
docker info >/dev/null 2>&1  || { echo "ERROR: docker daemon isn't running" >&2; exit 1; }

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "==> coturn ($COTURN_IMAGE) on turn:127.0.0.1:${TURN_PORT} (user pipecat:pipecat)"
echo "    relay ports 30000-30040/udp — foreground, ctrl-c to stop."
echo

# --external-ip as an ARG (not in the conf): supplying it suppresses the
# image's auto-detection, which would otherwise advertise the container IP —
# unreachable from the macOS host. Docker Desktop can't host-network, so
# allocations are advertised on loopback and reached via the -p mappings.
exec docker run --rm --name "$CONTAINER_NAME" \
  -p "${TURN_PORT}:3478/udp" -p "${TURN_PORT}:3478/tcp" \
  -p "30000-30040:30000-30040/udp" \
  -v "$SCRIPT_DIR/turnserver.dev.conf:/etc/coturn/turnserver.conf:ro" \
  "$COTURN_IMAGE" --external-ip=127.0.0.1
