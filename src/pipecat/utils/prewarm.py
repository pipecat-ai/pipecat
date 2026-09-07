#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Background loading of Pipecat's deferred third-party imports.

NLTK costs a few hundred milliseconds to import, reaching scikit-learn through
its optional classifier backends, but is needed only once text starts flowing.
It is imported where it is used, so building a pipeline doesn't pay for it.

Warming loads it while the pipeline is connecting to its services, so the first
sentence boundary doesn't pay for it either. The connect path is mostly waiting
on the network, which is where the work fits.
"""

from loguru import logger


def warm_deferred_imports() -> None:
    """Load the deferred third-party imports.

    Blocking and CPU-bound, so callers on an event loop should run it in a
    thread. Repeat calls are cheap. A module that fails to load is left for its
    point of use to report.
    """
    try:
        # Loaded by match_endofsentence(), first called on the opening bot turn.
        from pipecat.utils.string import _sent_tokenizer

        _sent_tokenizer()
    except Exception as e:
        logger.trace(f"Could not warm the NLTK sentence tokenizer: {e}")
