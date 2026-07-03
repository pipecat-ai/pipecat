#!/usr/bin/env python3
"""
Regenerate service configurations in _configs.py.

This is a convenience script that runs the config generator.
Similar to update_imports.py but for service configs.

Usage:
    uv run scripts/cli/configs/update_configs.py           # Update _configs.py
    uv run scripts/cli/configs/update_configs.py --preview # Preview without updating

Most callers should use the top-level ``scripts/cli/update_registry.py``, which
regenerates both _imports.py and _configs.py.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """Run the config generator."""
    parser = argparse.ArgumentParser(
        description="Generate or preview Pipecat service configurations"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview configurations without updating _configs.py",
    )
    args = parser.parse_args()

    script_path = Path(__file__).parent / "config_generator.py"

    # Build command
    cmd = ["uv", "run", str(script_path)]
    if args.preview:
        cmd.append("--preview")

    if not args.preview:
        print("Regenerating service configurations...")
        print("=" * 80)

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        if not args.preview:
            print("\n" + "=" * 80)
            print("✅ Service configurations updated successfully!")
            print("\nUpdated file:")
            print("  src/pipecat/cli/registry/_configs.py")
    else:
        if not args.preview:
            print("\n" + "=" * 80)
            print("❌ Failed to update service configurations")
        sys.exit(1)


if __name__ == "__main__":
    main()
