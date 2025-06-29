#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the script directory to ensure we're running from the right place
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔍 Running pre-commit checks..."

# Change to project root
cd "$PROJECT_ROOT"

# 1. Format check
echo "📝 Checking code formatting..."
if ! NO_COLOR=1 ruff format --diff --check; then
    echo -e "${RED}❌ Code formatting issues found. Run 'ruff format' to fix.${NC}"
    exit 1
fi

# 2. Lint check
echo "🔍 Running linter..."
if ! ruff check; then
    echo -e "${RED}❌ Linting issues found.${NC}"
    exit 1
fi

# 3. Custom docstring validation
echo "📚 Validating class docstrings..."
python3 "$PROJECT_ROOT/scripts/validate_docstrings.py"
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Docstring validation failed.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All pre-commit checks passed!${NC}"