#!/usr/bin/env bash
# start_voice_chatbot.sh — single-command launcher for the full multi-session stack
# 1. Docker daemon check  → Kokoro TTS (port 8880) & coturn (ports 3478/5349)
# 2. Python backend       → run_multisession_ws_patch.py (port 27880)
# 3. Cloudflared tunnel   → exposes backend publicly
# 4. Next.js UI           → npm run dev (port 3000)
# All services run in background; Ctrl-C stops everything.

set -euo pipefail

##############################################
# Helpers                                    #
##############################################
log() { echo -e "[\033[1;34mINFO\033[0m] $*"; }
err() { echo -e "[\033[0;31mERROR\033[0m] $*" >&2; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { err "'$1' is required but not installed."; exit 1; }
}

##############################################
# Environment / paths                        #
##############################################
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$ROOT_DIR"

# Python venv
if [[ -f "venv2/bin/activate" ]]; then
  source venv2/bin/activate
else
  err "Python venv (venv2) not found. Run: python -m venv venv2 && source venv2/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# Required commands
require_cmd docker
require_cmd cloudflared
require_cmd npm

##############################################
# Check Docker daemon                        #
##############################################
if ! docker info >/dev/null 2>&1; then
  err "Docker daemon is not running. Please start Docker Desktop or dockerd."
  exit 1
fi

##############################################
# Start Kokoro TTS                           #
##############################################
KOKORO_IMAGE="ghcr.io/pipecat-ai/kokoro:latest"
KOKORO_NAME="pipecat-kokoro-tts"
if ! docker ps --format '{{.Names}}' | grep -q "^${KOKORO_NAME}$"; then
  log "Starting Kokoro TTS container…"
  docker run -d --name "$KOKORO_NAME" -p 8880:8080 "$KOKORO_IMAGE" >/dev/null
else
  log "Kokoro TTS already running."
fi

# Export Pipecat env pointing at Kokoro shim
export OPENAI_API_BASE="http://localhost:8880/v1"
export OPENAI_API_KEY="dummy"

##############################################
# Start coturn (TURN/STUN)                   #
##############################################
COTURN_IMAGE="instrumentisto/coturn"
COTURN_NAME="pipecat-coturn"
if ! docker ps --format '{{.Names}}' | grep -q "^${COTURN_NAME}$"; then
  log "Starting coturn server…"
  docker run -d --name "$COTURN_NAME" \
    -p 3478:3478 -p 3478:3478/udp \
    -p 5349:5349 -p 5349:5349/udp \
    -e TURN_REALM=pipecat \
    -e TURN_USER=alex:supersecret "$COTURN_IMAGE" >/dev/null
else
  log "coturn already running."
fi

##############################################
# Trap to clean up                           #
##############################################
cleanup() {
  log "Shutting down services…"
  pkill -P $$ || true
  docker stop "$KOKORO_NAME" "$COTURN_NAME" >/dev/null 2>&1 || true
  exit 0
}
trap cleanup INT TERM

##############################################
# Launch Pipecat backend                     #
##############################################
log "Launching Pipecat multi-session backend… (http://localhost:27880)"
python src/pipecat/examples/run_multisession_ws_patch.py --host 0.0.0.0 --port 27880 &

##############################################
# Launch Cloudflared tunnel                  #
##############################################
log "Starting Cloudflared tunnel (pipecat-tunnel)…"
cloudflared tunnel run pipecat-tunnel &

##############################################
# Launch Next.js UI                          #
##############################################
log "Starting Next.js dev UI… (http://localhost:3000)"
cd examples/foundational/voice-chat-ui
npm run dev &
cd "$ROOT_DIR"

##############################################
# Wait                                        #
##############################################
log "All services started. Press Ctrl-C to stop."
wait
