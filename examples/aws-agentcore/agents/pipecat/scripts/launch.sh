#!/bin/bash

# Script to dynamically read all variables from .env file and launch agentcore
YAML_FILE=".bedrock_agentcore.yaml"
SERVER_ENV_FILE="../../server/.env"

###############################################
# STEP 1 — Launch the new agent
###############################################

# Check if the local .env file exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found in current directory"
    echo "Please create a .env file with your environment variables"
    exit 1
fi

# Start building the agentcore launch command
LAUNCH_CMD="agentcore launch --auto-update-on-conflict"

echo "Loading environment variables from .env file..."

# Read each line from .env file and process it
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    # Check if line contains an equals sign (valid env var format)
    if [[ "$line" =~ ^[^=]+=[^=]*$ ]]; then
        # Extract variable name and value
        VAR_NAME=$(echo "$line" | cut -d'=' -f1 | xargs)
        VAR_VALUE=$(echo "$line" | cut -d'=' -f2- | xargs)

        # Skip PIPECAT_LOCAL_DEV variable
        if [[ "$VAR_NAME" == "PIPECAT_LOCAL_DEV" ]]; then
            echo "  Skipping: $VAR_NAME (ignored for deployment)"
            continue
        fi

        # Skip if variable name or value is empty
        if [[ -n "$VAR_NAME" && -n "$VAR_VALUE" ]]; then
            # Add to launch command
            LAUNCH_CMD="$LAUNCH_CMD --env $VAR_NAME=$VAR_VALUE"
            echo "  Added: $VAR_NAME"
        fi
    fi
done < ".env"

# Check if any environment variables were added
if [[ "$LAUNCH_CMD" == "agentcore launch --auto-update-on-conflict" ]]; then
    echo "Warning: No valid environment variables found in .env file"
    echo "Make sure your .env file contains variables in the format: KEY=value"
    exit 1
fi

# Execute the command
echo ""
echo "Executing: $LAUNCH_CMD"
eval "$LAUNCH_CMD"


###############################################
# STEP 2 — Extract AGENT ARN from YAML
###############################################
if [ ! -f "$YAML_FILE" ]; then
    echo "ERROR: $YAML_FILE not found!"
    exit 1
fi

# Extracts: agent_arn: <value>
AGENT_ARN=$(grep -E "^\s*agent_arn:" "$YAML_FILE" | awk '{print $2}')

# Wait until it exists
while [ -z "$AGENT_ARN" ]; do
    sleep 0.2
    AGENT_ARN=$(grep -E "^\s*agent_arn:" "$YAML_FILE" | awk '{print $2}')
done

echo "Extracted Agent ARN: $AGENT_ARN"

###############################################
# STEP 3 — Update server .env
###############################################
if [ ! -f "$SERVER_ENV_FILE" ]; then
    echo "ERROR: $SERVER_ENV_FILE not found!"
    exit 1
fi

# If AGENT_RUNTIME_ARN already exists → replace
# If not → append
if grep -q "^AGENT_RUNTIME_ARN=" "$SERVER_ENV_FILE"; then
    sed -i.bak "s|^AGENT_RUNTIME_ARN=.*|AGENT_RUNTIME_ARN=$AGENT_ARN|" "$SERVER_ENV_FILE"
else
    echo "AGENT_RUNTIME_ARN=$AGENT_ARN" >> "$SERVER_ENV_FILE"
fi

echo ".env updated successfully!"
echo "AGENT_RUNTIME_ARN is now set to:"
echo "$AGENT_ARN"