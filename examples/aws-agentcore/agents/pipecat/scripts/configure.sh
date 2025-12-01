#!/bin/bash

# Script to configure the bot, patch Dockerfile and sync AGENT_RUNTIME_ARN

DOCKERFILE=".bedrock_agentcore/pipecat_agent/Dockerfile"
TARGET_LINE="RUN uv pip install -r requirements.txt"
# Extra dependencies needed by SmallWebRTC
INSERT_LINE="RUN apt update && apt install -y libgl1 libglib2.0-0 && apt clean"

###############################################
# STEP 1 — Configure agentcore
# Already configuring to use Docker as it is required by Pipecat
###############################################
agentcore configure -e pipecat-agent.py --deployment-type container --container-runtime docker

###############################################
# STEP 2 — Wait until Dockerfile exists
###############################################
while [ ! -s "$DOCKERFILE" ]; do
    sleep 0.2
done

###############################################
# STEP 3 — Patch Dockerfile
###############################################
cp "$DOCKERFILE" "$DOCKERFILE.bak"

awk -v target="$TARGET_LINE" -v insert="$INSERT_LINE" '
{
    print $0
    if ($0 ~ target) {
        print insert
    }
}
' "$DOCKERFILE.bak" > "$DOCKERFILE"

echo "Dockerfile patched successfully!"
