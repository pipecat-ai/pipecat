
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Running ruff format..."
ruff format "$PROJECT_ROOT"
echo "Running ruff check..."
ruff check --fix "$PROJECT_ROOT"
