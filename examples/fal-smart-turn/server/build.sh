#!/bin/bash
set -e

VERSION="0.1"
DOCKER_USERNAME=""
AGENT_NAME="pcc-smart-turn"

# Build the Docker image with the correct context
echo "Building Docker image..."
docker build --platform=linux/arm64 -t "$DOCKER_USERNAME/$AGENT_NAME:$VERSION" -t "$DOCKER_USERNAME/$AGENT_NAME:latest" .

# Push the Docker images
echo "Pushing Docker image $DOCKER_USERNAME/$AGENT_NAME:$VERSION..."
docker push "$DOCKER_USERNAME/$AGENT_NAME:$VERSION"

echo "Pushing Docker image $DOCKER_USERNAME/$AGENT_NAME:latest..."
docker push "$DOCKER_USERNAME/$AGENT_NAME:latest"

echo "Successfully built and pushed $DOCKER_USERNAME/$AGENT_NAME:$VERSION and $DOCKER_USERNAME/$AGENT_NAME:latest"