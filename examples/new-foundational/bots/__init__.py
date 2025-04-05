#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import pkgutil
from pathlib import Path


# Find all bot modules in the bots directory
def get_available_bots():
    """Automatically discover all bot modules in the bots package.

    Returns a dictionary of bot_name: module_name.
    """
    bot_modules = {}

    # Get the current directory
    package_path = Path(__file__).parent

    # Find all Python modules in this directory
    for _, module_name, is_pkg in pkgutil.iter_modules([str(package_path)]):
        # Skip __init__ and any other special files
        if module_name.startswith("_"):
            continue

        # The bot name is the module name with underscores converted to hyphens
        bot_name = module_name.replace("_", "-")
        bot_modules[bot_name] = f"bots.{module_name}"

    return bot_modules


# Dict of available example bots
EXAMPLES = get_available_bots()
