#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Network bus implementations for distributed workers.

Each adapter has its own optional dependency. Imports are lazy so the
package can be loaded with only the extras you need; importing a specific
bus without its extra raises a clear error from that submodule.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipecat.bus.network.pgmq import PgmqBus
    from pipecat.bus.network.redis import RedisBus

__all__ = ["PgmqBus", "RedisBus"]


def __getattr__(name: str):
    if name == "PgmqBus":
        from pipecat.bus.network.pgmq import PgmqBus

        return PgmqBus
    if name == "RedisBus":
        from pipecat.bus.network.redis import RedisBus

        return RedisBus
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
