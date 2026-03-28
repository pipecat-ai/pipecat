#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import warnings

from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService, Params

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Types in pipecat.services.aws_nova_sonic are deprecated. "
        "Please use the equivalent types from "
        "pipecat.services.aws.nova_sonic.llm instead.",
        DeprecationWarning,
        stacklevel=2,
    )

__all__ = [
    "AWSNovaSonicLLMService",
    "Params",
]
