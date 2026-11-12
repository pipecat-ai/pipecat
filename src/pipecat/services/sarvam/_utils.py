#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Helpers shared by the Sarvam services."""

from dataclasses import fields
from typing import Any


def reject_settings_kwargs(service: Any, kwargs: dict[str, Any]) -> None:
    """Raise if a constructor kwarg is a field of the service's ``Settings``.

    Extra kwargs travel up to ``BaseObject``, which drops anything it doesn't
    know. A setting passed directly, such as ``voice="anushka"``, would
    otherwise be ignored without a word and the default used instead.

    Args:
        service: The service being constructed.
        kwargs: The ``**kwargs`` its ``__init__`` received.

    Raises:
        TypeError: If any kwarg names a ``Settings`` field.
    """
    service_name = type(service).__name__
    settings_fields = {setting.name for setting in fields(service.Settings)}
    direct_settings = sorted(settings_fields.intersection(kwargs))
    if direct_settings:
        names = ", ".join(direct_settings)
        raise TypeError(
            f"{names} must be passed via "
            f"settings={service_name}.Settings(...), not as constructor kwargs"
        )
