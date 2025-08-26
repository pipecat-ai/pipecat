#!/bin/bash
set -e

VERSION="0.1"
DOCKER_USERNAME="your_username"
AGENT_NAME="quickstart"

# Function to check if buildx is available
check_buildx() {
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed or not in PATH"
        echo "   Please install Docker Desktop or Docker Engine"
        echo "   Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi

    if ! docker buildx version &> /dev/null; then
        echo "❌ Docker buildx is not available"
        echo "   buildx is required for cross-platform ARM64 builds"
        echo ""
        echo "Solutions:"
        echo "  • Install Docker Desktop (includes buildx)"
        echo "  • Update Docker to version 19.03+ and enable buildx"
        echo "  • Run: docker buildx install"
        echo ""
        echo "Visit: https://docs.docker.com/buildx/working-with-buildx/"
        exit 1
    fi

    echo "✅ Docker buildx is available"
}

# Function to ensure builder is set up
setup_builder() {
    # Check if we have a builder that supports our target platform
    if ! docker buildx inspect --builder default | grep -q "linux/arm64"; then
        echo "Setting up multi-platform builder..."
        docker buildx create --name multiarch --driver docker-container --use --bootstrap 2>/dev/null || {
            echo "⚠️  Using default builder (may be slower on x86 machines)"
            docker buildx use default
        }
    fi
}

# Main build process
main() {
    echo "Building Docker image for Pipecat Cloud (ARM64)..."

    # Check prerequisites
    check_buildx
    setup_builder

    # Build the Docker image with buildx (handles cross-platform)
    echo "Building $DOCKER_USERNAME/$AGENT_NAME:$VERSION..."

    # Use buildx with --load to build and load into local Docker
    if ! docker buildx build \
        --platform=linux/arm64 \
        -t "$DOCKER_USERNAME/$AGENT_NAME:$VERSION" \
        -t "$DOCKER_USERNAME/$AGENT_NAME:latest" \
        --load \
        .; then
        echo "❌ Build failed"
        echo ""
        echo "Common issues:"
        echo "  • Missing Dockerfile in current directory"
        echo "  • Build errors in your application code"
        echo "  • Insufficient disk space"
        echo ""
        echo "On Intel/x86 machines: builds may be slower due to emulation"
        exit 1
    fi

    echo "✅ Build completed successfully"

    # Push the Docker images
    echo ""
    echo "Pushing images to registry..."

    echo "Pushing $DOCKER_USERNAME/$AGENT_NAME:$VERSION..."
    if ! docker push "$DOCKER_USERNAME/$AGENT_NAME:$VERSION"; then
        echo "❌ Push failed for version tag"
        echo "   Make sure you're logged in: docker login"
        exit 1
    fi

    echo "Pushing $DOCKER_USERNAME/$AGENT_NAME:latest..."
    if ! docker push "$DOCKER_USERNAME/$AGENT_NAME:latest"; then
        echo "❌ Push failed for latest tag"
        echo "   Make sure you're logged in: docker login"
        exit 1
    fi

    echo ""
    echo "🎉 Successfully built and pushed:"
    echo "   📦 $DOCKER_USERNAME/$AGENT_NAME:$VERSION"
    echo "   📦 $DOCKER_USERNAME/$AGENT_NAME:latest"
    echo ""
    echo "Next steps:"
    echo "   1. Update your pcc-deploy.toml with the image name"
    echo "   2. Run: pcc deploy"
}

# Run main function
main "$@"
