#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The HTTP client family the installed OpenAI SDK is built on.

openai < 3 builds on httpx and openai >= 3 on httpx2, and each installs only its
own. Tests that construct SDK-facing objects — requests, responses, timeouts —
take them from here so they run against whichever family is installed.
"""

import importlib

from openai import DefaultAsyncHttpxClient

#: The SDK's own async client class.
ASYNC_CLIENT = DefaultAsyncHttpxClient.__mro__[1]

#: The module that class comes from, holding the matching ``Request``,
#: ``Response``, ``Timeout`` and exception types.
http = importlib.import_module(ASYNC_CLIENT.__module__.split(".")[0])
