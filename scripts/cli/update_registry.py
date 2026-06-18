#!/usr/bin/env python3
"""
Regenerate all auto-generated registry files.

This convenience script updates both imports and configs.

Usage:
    uv run scripts/update_registry.py           # Update both _imports.py and _configs.py
    uv run scripts/update_registry.py --preview # Preview without updating
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """Run both import and config generators."""
    parser = argparse.ArgumentParser(
        description="Generate or preview all Pipecat registry files (imports + configs)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview without updating files",
    )
    args = parser.parse_args()

    scripts_dir = Path(__file__).parent
    update_imports = scripts_dir / "imports" / "update_imports.py"
    update_configs = scripts_dir / "configs" / "update_configs.py"

    if not args.preview:
        print("🔄 Regenerating registry files...")
        print("=" * 80)
        print()

    # Run imports update
    print("📦 Updating service imports...")
    cmd_imports = ["uv", "run", str(update_imports)]
    if args.preview:
        cmd_imports.append("--preview")
        print("\n--- IMPORTS PREVIEW ---\n")

    result_imports = subprocess.run(cmd_imports, capture_output=False)

    if result_imports.returncode != 0:
        print("\n❌ Failed to update imports")
        sys.exit(1)

    if not args.preview:
        print()
        print("-" * 80)
        print()

    # Run configs update
    print("⚙️  Updating service configurations...")
    cmd_configs = ["uv", "run", str(update_configs)]
    if args.preview:
        cmd_configs.append("--preview")
        print("\n--- CONFIGS PREVIEW ---\n")

    result_configs = subprocess.run(cmd_configs, capture_output=False)

    if result_configs.returncode != 0:
        print("\n❌ Failed to update configs")
        sys.exit(1)

    # Success!
    if not args.preview:
        print()
        print("=" * 80)
        print("✅ All registry files updated successfully!")
        print("\nUpdated files:")
        print("  • src/pipecat/cli/registry/_imports.py")
        print("  • src/pipecat/cli/registry/_configs.py")
        print()


if __name__ == "__main__":
    main()
