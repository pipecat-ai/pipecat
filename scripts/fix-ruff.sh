
#!/bin/bash

echo "Running ruff format..."
ruff format src/
ruff check --fix src/
