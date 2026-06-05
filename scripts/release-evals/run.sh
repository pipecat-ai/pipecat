#!/usr/bin/env sh
#
# Release evals: spawn each agent in manifest.yaml with the eval transport and
# run its scenarios against it (via `pipecat eval suite`). Extra args are
# forwarded, e.g.:
#
#   ./run.sh                       # everything
#   ./run.sh -p voice-openai       # only matching agents
#   ./run.sh -s simple_math        # only the simple_math scenario
#   ./run.sh -c 8 -a               # 8 at a time, record audio
#
set -e
here="$(cd "$(dirname "$0")" && pwd)"
exec uv run python -m pipecat.evals suite \
  "$here/manifest.yaml" \
  --runs-dir "$here/test-runs/$(date +%Y%m%d_%H%M%S)" \
  "$@"
