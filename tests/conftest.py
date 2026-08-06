#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared pytest configuration for the Pipecat test suite."""

import dotenv


def pytest_configure(config):
    """Keep a developer's ``.env`` out of the test session.

    Modules that call ``load_dotenv()`` at import scope would otherwise pull the
    repository's ``.env`` into ``os.environ`` while pytest collects the suite,
    before any test runs. Tests would then see whichever variables that
    developer happens to have set, and behave differently than they do in CI.

    Collection imports test modules after this hook, so a module binding the
    name with ``from dotenv import load_dotenv`` picks up the stub.
    """
    dotenv.load_dotenv = lambda *args, **kwargs: False
