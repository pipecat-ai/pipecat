#!/usr/bin/env sh
#
# Turn-completion evals: spawn the bot in manifest.yaml once per model and
# scenario (via `pipecat eval suite`) and score the LLM's turn-completion
# markers. Output goes to test-runs/<name>/ (set by the manifest's runs_dir).
# Extra args forward, e.g.:
#
#   ./run.sh -n baseline                  # every model, every scenario
#   ./run.sh -n baseline -p openai        # only matching entries
#   ./run.sh -s cutoff/cutoff_preposition # one scenario
#   ./run.sh -n rates -r 3                # three attempts per run
#   TURN_COMPLETION_PROMPT=v3 ./run.sh -n v3   # a prompt variant from prompts/
#
# A marker or reply arrives within seconds when the model follows the
# protocol, so an expectation without its own within_ms times out after 30 s
# (-t 30); a -t of your own overrides it.
#
set -e
here="$(cd "$(dirname "$0")" && pwd)"
exec uv run python -m pipecat.evals suite "$here/manifest.yaml" -t 30 "$@"
