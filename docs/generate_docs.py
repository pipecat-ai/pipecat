#!/usr/bin/env python3

import shutil
import subprocess
from pathlib import Path


def run_command(command: list[str]) -> None:
    """Run a command and exit if it fails."""
    print(f"Running: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Command failed: {' '.join(command)}")
        print(f"Error: {e}")


def main():
    docs_dir = Path(__file__).parent
    project_root = docs_dir.parent

    # Install documentation requirements
    requirements_file = docs_dir / "requirements.txt"
    run_command(["pip", "install", "-r", str(requirements_file)])

    # Install core package
    run_command(["pip", "install", "-e", "."])

    # Install all service dependencies
    services = [
        "anthropic",
        "assemblyai",
        "aws",
        "azure",
        "canonical",
        "cartesia",
        # "daily",
        "deepgram",
        "elevenlabs",
        "fal",
        "fireworks",
        "gladia",
        "google",
        "grok",
        "groq",
        "langchain",
        # "livekit",
        "lmnt",
        "moondream",
        "nim",
        "noisereduce",
        "openai",
        "openpipe",
        "playht",
        "silero",
        "soundfile",
        "websocket",
        "whisper",
    ]

    extras = ",".join(services)
    try:
        run_command(["pip", "install", "-e", f".[{extras}]"])
    except Exception as e:
        print(f"Warning: Some dependencies failed to install: {e}")

    # Clean old files
    api_dir = docs_dir / "api"
    build_dir = docs_dir / "_build"
    for dir in [api_dir, build_dir]:
        if dir.exists():
            shutil.rmtree(dir)

    # Generate API documentation
    run_command(
        [
            "sphinx-apidoc",
            "-f",  # Force overwrite
            "-e",  # Put each module on its own page
            "-M",  # Put module documentation before submodule
            "--no-toc",  # Don't generate modules.rst (cleaner structure)
            "-o",
            str(api_dir),  # Output directory
            str(project_root / "src/pipecat"),
            # Exclude problematic files and directories
            "**/processors/gstreamer/*",  # Optional gstreamer
            "**/transports/network/*",  # Pydantic issues
            "**/transports/services/*",  # Pydantic issues
            "**/transports/local/*",  # Optional dependencies
            "**/services/to_be_updated/*",  # Exclude to_be_updated package
            "**/*test*",  # Test files
        ]
    )

    # Build HTML documentation
    run_command(["sphinx-build", "-b", "html", str(docs_dir), str(build_dir / "html")])

    print("\nDocumentation generated successfully!")
    print(f"HTML docs: {build_dir}/html/index.html")


if __name__ == "__main__":
    main()
