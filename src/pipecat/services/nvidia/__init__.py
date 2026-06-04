#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

# Suppress verbose gRPC C-core logging (fork handlers, abseil warnings) unless
# the user has explicitly configured it.
if "GRPC_VERBOSITY" not in os.environ:
    os.environ["GRPC_VERBOSITY"] = "ERROR"
