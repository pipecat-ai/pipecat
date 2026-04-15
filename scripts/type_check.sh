#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}/.."

uv sync --extra vonage-video-connector --extra aws --extra aws-nova-sonic
uv pip install mypy

files=(
    "$SCRIPT_DIR/../tests/test_vonage_video_connector.py"
    "$SCRIPT_DIR/../src/pipecat/transports/vonage/video_connector.py"
    "$SCRIPT_DIR/../src/pipecat/transports/vonage/client.py"
    "$SCRIPT_DIR/../src/pipecat/transports/vonage/utils.py"
    "$SCRIPT_DIR/../examples/foundational/04c-transports-vonage-video-connector.py"
)
for file in "${files[@]}"; do
    echo -e "\033[0;32mType checking file: $file\033[0m"

    uv run mypy "$file" --strict
    if [ $? -ne 0 ]; then
        echo -e "\033[0;31mType checking failed for $file\033[0m"
        exit 1
    else
        echo -e "\033[0;32mType checking passed for $file\033[0m"
    fi
done
