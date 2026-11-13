#!/usr/bin/env python3
"""
Check service registry for completeness and consistency.

This script helps identify services that need configs or imports,
making it easier to maintain the registry as Pipecat evolves.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pipecat.cli.registry import ServiceLoader, ServiceRegistry


def main():
    """Check registry and report issues."""
    print("🔍 Checking service registry...")
    print()

    # Check for missing configs and imports
    missing = ServiceLoader.get_missing_services()

    has_issues = False

    if missing["missing_configs"]:
        has_issues = True
        print("❌ Services missing configs:")
        for service in missing["missing_configs"]:
            print(f"   - {service}")
        print()
        print("   Define them in src/pipecat/cli/registry/service_metadata.py, then run")
        print("   `uv run scripts/cli/update_registry.py` to regenerate _configs.py")
        print()

    if missing["missing_imports"]:
        has_issues = True
        print("❌ Services missing imports:")
        for service in missing["missing_imports"]:
            print(f"   - {service}")
        print()
        print("   Define them in src/pipecat/cli/registry/service_metadata.py, then run")
        print("   `uv run scripts/cli/update_registry.py` to regenerate _imports.py")
        print()

    # Count services
    service_counts = {
        "WebRTC Transports": len(ServiceRegistry.WEBRTC_TRANSPORTS),
        "Telephony Transports": len(ServiceRegistry.TELEPHONY_TRANSPORTS),
        "STT Services": len(ServiceRegistry.STT_SERVICES),
        "LLM Services": len(ServiceRegistry.LLM_SERVICES),
        "TTS Services": len(ServiceRegistry.TTS_SERVICES),
        "Realtime Services": len(ServiceRegistry.REALTIME_SERVICES),
    }

    print("📊 Service counts:")
    for category, count in service_counts.items():
        print(f"   {category}: {count}")
    print()

    if not has_issues:
        print("✅ All services have configs and imports!")
        return 0
    else:
        print("⚠️  Please add missing configs and imports.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
