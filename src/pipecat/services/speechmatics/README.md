# Speechmatics STT Service

NOTES TO GO HERE!

- You need to have the `SPEECHMATICS_API_KEY` environment variable set.

```shell
# setup
python -m venv .venv
source .venv/bin/activate

# install deps
pip install -r dev-requirements.txt
pip install -e .
pip install -e ".[daily,deepgram,openai,silero,speechmatics]"

# module
ruff check --fix --select I001 src/pipecat/services/speechmatics
ruff check --fix src/pipecat/services/speechmatics
ruff format src/pipecat/services/speechmatics

# example
ruff check --fix --select I001 examples/foundational/07a-interruptible-speechmatics.py
ruff check --fix examples/foundational/07a-interruptible-speechmatics.py
ruff format examples/foundational/07a-interruptible-speechmatics.py

# run example
pip install -r examples/foundational/requirements.txt
python examples/foundational/07a-interruptible-speechmatics.py
```
