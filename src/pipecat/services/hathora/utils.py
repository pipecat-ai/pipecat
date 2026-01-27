#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Utilities and types for [Hathora-hosted](https://models.hathora.dev) voice services."""

from dataclasses import dataclass


@dataclass
class ConfigOption:
    """Extra configuration option passed into model_config for Hathora (if supported by model).

    Args:
        name: Name of the configuration option.
        value: Value of the configuration option.
    """

    name: str
    value: str
