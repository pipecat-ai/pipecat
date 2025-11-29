
#!/bin/bash

# Script to dynamically read all variables from .env file and launch agentcore

# Check if .env file exists
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