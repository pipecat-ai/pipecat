
#!/bin/bash

echo "Running ruff format..."
uv run ruff format "$PROJECT_ROOT"
echo "Running ruff check..."
uv run ruff check --fix "$PROJECT_ROOT"
