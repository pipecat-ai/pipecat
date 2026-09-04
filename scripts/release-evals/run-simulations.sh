#!/usr/bin/env sh
#
# Release simulations: spawn each bot in manifest-simulations.yaml with the
# eval transport and run its simulations against it (via `pipecat eval suite`).
# Output goes to test-runs/<timestamp>/ (set by the manifest's runs_dir). Extra
# args forward, e.g.:
#
#   ./run-simulations.sh                       # everything
#   ./run-simulations.sh -p restaurant         # only matching bots
#   ./run-simulations.sh -s order_pizza -r 5   # one simulation, five runs
#
set -e
here="$(cd "$(dirname "$0")" && pwd)"
exec uv run python -m pipecat.evals suite -d "$here/manifest-simulations.yaml" "$@"
