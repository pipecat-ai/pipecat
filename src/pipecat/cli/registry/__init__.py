#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Service registry for Pipecat services.

The registry consists of:
- service_metadata.py: SOURCE OF TRUTH - All service definitions
- service_loader.py: Logic for loading and validating services
- _configs.py: AUTO-GENERATED - Service initialization code
- _imports.py: AUTO-GENERATED - Import statements
"""

from .service_loader import ServiceLoader, extract_package_extra
from .service_metadata import BotType, ServiceDefinition, ServiceRegistry, ServiceType

__all__ = [
    "ServiceRegistry",
    "ServiceLoader",
    "ServiceDefinition",
    "ServiceType",
    "BotType",
    "extract_package_extra",
]
