#!/usr/bin/env sh
#
# Release evals: spawn each bot in manifest.yaml with the eval transport and
# run its scenarios against it (via `pipecat eval suite`). Output goes to
# test-runs/<timestamp>/ (set by the manifest's runs_dir). Extra args forward,
# e.g.:
#
#   ./run.sh                       # everything, scripted and simulated
#   ./run.sh -p voice-openai       # only matching bots
#   ./run.sh -s capital_question   # only the capital_question scenario
#   ./run.sh -k script             # every scripted scenario, no simulations
#   ./run.sh -k simulation         # every simulation, nothing scripted
#   ./run.sh -s order_pizza -r 5   # one simulation, five runs
#   ./run.sh -c 8 -a               # 8 at a time, record audio
#
set -e
here="$(cd "$(dirname "$0")" && pwd)"
exec uv run python -m pipecat.evals suite -d "$here/manifest.yaml" "$@"
