#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deprecation marker conventions for the Pipecat framework.

Every deprecation in Pipecat is emitted in one of three ways, all producing a
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

**Everything else — parameters, fields, module moves, behavior/value changes:**
the decorator cannot mark these, so warn with :func:`warn_deprecated`, never a
bare ``warnings.warn``. These do not get static-checker detection, but the
``.. deprecated::`` directive (below) still records them for documentation and
tooling. Pass the message as a string or f-string literal so
``tests/test_deprecation_markers.py`` can check it against the template, and set
``stacklevel`` so the warning names the caller's line.

In all cases, add a ``.. deprecated:: X.Y.Z`` directive to the docstring (for a
parameter, in its ``Args:`` / ``Parameters:`` entry). The directive is the
single source of truth that downstream tooling parses into a deprecation
registry, so its body follows a small grammar — a replacement clause naming the
target, or an explicit "No replacement." — enforced by
``tests/test_deprecation_markers.py``::

    .. deprecated:: 1.3.0
        Use :class:`~pipecat.pipeline.worker.PipelineWorker` instead.        # rename / use-existing
        Merged into :class:`LLMContext`.            # capability absorbed
        Moved to :mod:`pipecat.services.xai.llm`.   # module move
        No replacement.                             # nothing to migrate to

Prefer Sphinx cross-reference roles (``:class:``, ``:meth:``, ``:func:``,
``:attr:``, ``:mod:``) for the target — they encode its kind and resolve in
docs — but a backticked name is accepted.
"""

import re
import sys
import warnings

from typing_extensions import deprecated

__all__ = ["DEPRECATION_MESSAGE_RE", "deprecated", "warn_deprecated", "warn_deprecated_read"]

# The canonical deprecation message, for @deprecated and hand-written warnings
# alike. Kept consistent and parseable so the developer-facing message agrees
# with the docstring directive.
DEPRECATION_MESSAGE_RE = re.compile(
    r"^`(?P<subject>[^`]+)` is deprecated since (?P<version>\d+\.\d+\.\d+) "
    r"and will be removed in (?P<removal>\d+\.\d+\.\d+)\. "
    r"(?:Use (?P<replacement>.+) instead\.|No replacement\.)"
)


# Call sites already warned about by :func:`warn_deprecated`, identified by
# message and source location.
_warned_sites: set[tuple[str, str, int]] = set()


def warn_deprecated(message: str, stacklevel: int = 1) -> None:
    """Warn once per call site that something deprecated was used.

    Python's default filters hide ``DeprecationWarning`` unless the line it
    names is in ``__main__``, and many of Pipecat's deprecations are detected
    inside Pipecat, as a pipeline runs, where no line of the developer's code is
    on the stack. So the warning is raised under the ``always`` filter, which
    shows it wherever it is raised, and a per-site record keeps it to one
    warning per call site: a deprecated field read wherever its object travels,
    or a check on every frame, would otherwise repeat without bound.

    The record of warned sites lasts for the life of the process, so a test
    asserting on one of these warnings clears :data:`_warned_sites` first.

    Args:
        message: The warning message, following :data:`DEPRECATION_MESSAGE_RE`.
        stacklevel: As for :func:`warnings.warn`, counted from the caller: ``1``
            names the line that calls this function, ``2`` its caller.
    """
    try:
        frame = sys._getframe(stacklevel)
        site = (message, frame.f_code.co_filename, frame.f_lineno)
    except ValueError:
        site = (message, "", 0)
    if site in _warned_sites:
        return
    _warned_sites.add(site)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(message, DeprecationWarning, stacklevel=stacklevel + 1)


@deprecated(
    "`warn_deprecated_read` is deprecated since 1.13.0 and will be removed in 2.0.0. "
    "Use `warn_deprecated` instead."
)
def warn_deprecated_read(message: str) -> None:
    """Warn once per call site that a deprecated field was read.

    Call it from the ``__getattribute__`` that intercepts the read.

    .. deprecated:: 1.13.0
        Use :func:`warn_deprecated` instead, with ``stacklevel=2``.
        Will be removed in 2.0.0.

    Args:
        message: The warning message.
    """
    # Past this function, the decorator's wrapper, and __getattribute__ to the
    # line that read the field.
    warn_deprecated(message, stacklevel=4)
