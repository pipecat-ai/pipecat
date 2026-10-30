#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from types import SimpleNamespace

from pipecat.utils.tracing.service_decorators import _get_parent_service_context, _get_turn_context
from pipecat.utils.tracing.setup import is_tracing_available


class _TruthyBrokenTurnContextProxy:
    def __bool__(self):
        return True

    def get_turn_context(self):
        inner = None
        return inner.get_turn_context()


class _TruthyBrokenConversationContextProxy:
    def __bool__(self):
        return True

    def get_conversation_context(self):
        inner = None
        return inner.get_conversation_context()


def test_get_turn_context_handles_none_context():
    service = SimpleNamespace(_tracing_context=None)
    assert _get_turn_context(service) is None


def test_get_turn_context_handles_truthy_broken_proxy():
    service = SimpleNamespace(_tracing_context=_TruthyBrokenTurnContextProxy())
    assert _get_turn_context(service) is None


def test_get_parent_service_context_handles_none_context():
    service = SimpleNamespace(_tracing_context=None)
    value = _get_parent_service_context(service)
    if is_tracing_available():
        # Falls back to current context when no conversation context is available.
        assert value is not None
    else:
        assert value is None


def test_get_parent_service_context_handles_truthy_broken_proxy():
    service = SimpleNamespace(_tracing_context=_TruthyBrokenConversationContextProxy())
    value = _get_parent_service_context(service)
    if is_tracing_available():
        # Falls back to current context when conversation context getter crashes.
        assert value is not None
    else:
        assert value is None
