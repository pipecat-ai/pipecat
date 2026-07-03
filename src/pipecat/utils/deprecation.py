#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deprecation marker conventions for the Pipecat framework.

Every deprecation in Pipecat is emitted in one of two ways, both producing a
message that follows the same canonical template so it is machine-parseable
(see ``DEPRECATION_MESSAGE_RE``) and consistent for readers:

    `Subject` is deprecated since X.Y.Z and will be removed in A.B.C. Use `Replacement` instead.

where the removal version ``A.B.C`` is a concrete semantic version (e.g.
``2.0.0``) — we commit to the release that removes it rather than saying "a
future release" — and the second sentence is ``No replacement.`` when there is
nothing to migrate to, stated explicitly and never omitted. Additional
sentences may follow.

**Symbols — classes, functions, methods, properties:** mark with the PEP 702
``@deprecated`` decorator re-exported here. It emits the runtime
``DeprecationWarning`` automatically and lets type checkers and IDEs flag
usages statically (pyright's ``reportDeprecated``, mypy's ``deprecated`` error
code). Its argument must be a string literal — type checkers cannot display a
computed message — following the template above::

    @deprecated(
        "`OldService` is deprecated since 1.3.0 and will be removed in 2.0.0. "
        "Use `NewService` instead."
    )
    class OldService(NewService):
        \"\"\"Deprecated alias for :class:`NewService`.

        .. deprecated:: 1.3.0
            Use :class:`NewService` instead.
            Will be removed in 2.0.0.
        \"\"\"

**Everything else — parameters, module moves, behavior/value changes:** the
decorator cannot mark these, so emit a ``DeprecationWarning`` by hand with
``warnings.warn(..., DeprecationWarning)``. These do not get static-checker
detection, but the ``.. deprecated::`` directive (below) still records them for
documentation and tooling.

In all cases, add a ``.. deprecated:: X.Y.Z`` directive to the docstring (for a
parameter, in its ``Args:`` / ``Parameters:`` entry). The directive is the
single source of truth that downstream tooling parses into a deprecation
registry, so its body follows a small grammar — a replacement clause naming the
target, or an explicit "No replacement." — enforced by
``tests/test_deprecation_markers.py``::

    .. deprecated:: 1.3.0
        Use :class:`PipelineWorker` instead.        # rename / use-existing
        Merged into :class:`LLMContext`.            # capability absorbed
        Moved to :mod:`pipecat.services.xai.llm`.   # module move
        No replacement.                             # nothing to migrate to

Prefer Sphinx cross-reference roles (``:class:``, ``:meth:``, ``:func:``,
``:attr:``, ``:mod:``) for the target — they encode its kind and resolve in
docs — but a backticked name is accepted.
"""

import re

from typing_extensions import deprecated

__all__ = ["DEPRECATION_MESSAGE_RE", "deprecated"]

# The canonical @deprecated decorator message. Kept consistent and parseable so
# the developer-facing message agrees with the docstring directive.
DEPRECATION_MESSAGE_RE = re.compile(
    r"^`(?P<subject>[^`]+)` is deprecated since (?P<version>\d+\.\d+\.\d+) "
    r"and will be removed in (?P<removal>\d+\.\d+\.\d+)\. "
    r"(?:Use (?P<replacement>.+) instead\.|No replacement\.)"
)
