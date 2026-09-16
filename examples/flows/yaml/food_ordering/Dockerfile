# Pipecat Cloud image for the TypeSafe-driven food ordering bot.
#
# Build from the repository root, because the bot runs on this checkout of
# pipecat rather than a release from PyPI:
#
#   docker build --platform linux/arm64 \
#     -f examples/flows/yaml/food_ordering/Dockerfile -t food-ordering .
#
# `pipecat cloud deploy` from this directory does the same through
# pcc-deploy.toml, which points the build context at the repository root.
FROM dailyco/pipecat-base:latest-py3.13

# pipecat's version comes from git tags, which are not in the build context.
ARG PIPECAT_VERSION=1.10.1.dev0
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PIPECAT_AI=${PIPECAT_VERSION}

# Install this checkout of pipecat, with the extras the bot uses, into the
# base image's environment.
COPY pyproject.toml README.md LICENSE /pipecat/
COPY src /pipecat/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv/bin/python \
    "/pipecat[cartesia,daily,deepgram,runner,silero,typesafe,webrtc]"

# The bot and the flow it loads. The base image runs bot.py from /app.
COPY examples/flows/yaml/food_ordering/bot.py bot.py
COPY examples/flows/yaml/food_ordering/handlers.py handlers.py
COPY examples/flows/yaml/food_ordering/flow.yaml flow.yaml
