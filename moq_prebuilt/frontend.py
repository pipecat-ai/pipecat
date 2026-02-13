#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import logging
import os

from fastapi.staticfiles import StaticFiles

# Define possible paths to the client directory
base_dir = os.path.dirname(__file__)
possible_client_paths = [
    os.path.abspath(os.path.join(base_dir, "client")),  # in package
    os.path.abspath(os.path.join(base_dir, "..", "client")),  # in dev
]

client_dir = None

for path in possible_client_paths:
    logging.info(f"Checking MOQ client directory: {path}")
    if os.path.isdir(path):
        client_dir = path
        break

if not client_dir:
    logging.error("MOQ prebuilt client not found in any of the expected locations.")
    raise RuntimeError("MOQ prebuilt client not found.")

MOQPrebuiltUI = StaticFiles(directory=client_dir, html=True)
