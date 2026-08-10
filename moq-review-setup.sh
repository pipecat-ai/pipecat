#!/usr/bin/env bash
#
# TEMPORARY — review scaffolding for the MoQ client/relay + direct-mode PRs.
# Delete once the branches merge.
#
# Sets up a machine to review and run this branch of pipecat together with
# the pipecat-client-web-transports fix it depends on, then prints how to
# run the example bot against the review relay.
#
#   ./moq-review-setup.sh
#
# What it does:
#   1. pipecat (this repo): checks out the review branch and syncs deps.
#   2. pipecat-client-web-transports: fetches the subscribe-gating fix
#      (PR pipecat-ai/pipecat-client-web-transports#167) into a sibling
#      checkout and builds it. Needed to REVIEW that PR and for MoQ
#      direct-mode testing (see MOQ_DIRECT_DEV.md); the basic client-mode
#      example below runs fine on published packages without it.
#   3. Prints the bot command + browser URL.
#
# Env overrides:
#   PIPECAT_BRANCH  pipecat branch to review
#                   (default: vp-claude-moq-client-js-python-0dff25-rebased)
#   WT_DIR          pipecat-client-web-transports checkout
#                   (default: ../pipecat-client-web-transports)
#   WT_REMOTE       where to fetch the web-transports fix from (default:
#                   origin; the PR head also lives on the kixelated fork —
#                   WT_REMOTE=https://github.com/kixelated/pipecat-client-web-transports.git)
#   WT_BRANCH       web-transports branch (default: claude/moq-wait-for-bot-broadcast)
#   SKIP_WT=1       skip the web-transports setup entirely
#   RELAY_URL       relay for the example (default: https://relay.vanessa-dev.com/anon)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPECAT_BRANCH="${PIPECAT_BRANCH:-vp-claude-moq-client-js-python-0dff25-rebased}"
WT_DIR="${WT_DIR:-$SCRIPT_DIR/../pipecat-client-web-transports}"
WT_REMOTE="${WT_REMOTE:-origin}"
WT_BRANCH="${WT_BRANCH:-claude/moq-wait-for-bot-broadcast}"
RELAY_URL="${RELAY_URL:-https://relay.vanessa-dev.com/anon}"

echo "==> [1/3] pipecat: branch + deps"
cd "$SCRIPT_DIR"
current="$(git branch --show-current)"
if [[ "$current" != "$PIPECAT_BRANCH" ]]; then
    git fetch origin "$PIPECAT_BRANCH"
    git checkout "$PIPECAT_BRANCH"
fi
uv sync --group dev --all-extras --no-extra gstreamer --no-extra local
echo "    moq-ffi: $(uv run python -c 'import importlib.metadata as m; print(m.version("moq-ffi"))')"
echo "    (the relay must run the matching moq-relay release batch, or the"
echo "     RTVI transcript stream is silently dropped)"
echo

echo "==> [2/3] pipecat-client-web-transports: subscribe-gating fix (#167)"
if [[ "${SKIP_WT:-0}" == "1" ]]; then
    echo "    skipped (SKIP_WT=1)."
elif [[ ! -d "$WT_DIR/.git" ]]; then
    echo "    checkout not found at $WT_DIR — clone it or set WT_DIR. Skipping."
else
    git -C "$WT_DIR" fetch "$WT_REMOTE" "$WT_BRANCH"
    git -C "$WT_DIR" checkout -B "$WT_BRANCH" FETCH_HEAD
    (cd "$WT_DIR" && npm install)
    (cd "$WT_DIR/transports/moq-transport" && npm run build)
    echo "    built @pipecat-ai/moq-transport from $WT_BRANCH."
fi
echo

echo "==> [3/3] Ready. Run the example bot (client mode, dials the relay):"
echo
for key in DEEPGRAM_API_KEY OPENAI_API_KEY CARTESIA_API_KEY; do
    if ! grep -q "^${key}=" "$SCRIPT_DIR/.env" 2>/dev/null; then
        echo "    NOTE: $key not found in .env — the bot needs it."
    fi
done
cat <<EOF

    uv run python examples/transports/transports-moq.py -t moq \\
        --moq-connect $RELAY_URL

    Then open http://localhost:7860/client/ and pick "Media over QUIC".

    Direct mode (bot-per-browser, no /start; needs the prebuilt client from
    source): see MOQ_DIRECT_DEV.md, using --moq-connect $RELAY_URL
EOF
