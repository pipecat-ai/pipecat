#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base AI service implementation.

Provides the foundation for all AI services in the Pipecat framework, including
model management, settings handling, and frame processing lifecycle methods.
"""

from typing import Any, AsyncGenerator, Dict

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import ServiceSettings


class AIService(FrameProcessor):
    """Base class for all AI services.

    Provides common functionality for AI services including model management,
    settings handling, session properties, and frame processing lifecycle.
    Subclasses should implement specific AI functionality while leveraging
    this base infrastructure.
    """

    def __init__(self, settings: ServiceSettings | None = None, **kwargs):
        """Initialize the AI service.

        Args:
            settings: The runtime-updatable settings for the AI service.
            **kwargs: Additional arguments passed to the parent FrameProcessor.
        """
        super().__init__(**kwargs)
        self._settings: ServiceSettings = (
            settings
            # Here in case subclass doesn't implement more specific settings
            # (which hopefully should be rare)
            or ServiceSettings()
        )
        self._sync_model_name_to_metrics()
        self._session_properties: Dict[str, Any] = {}
        self._tracing_enabled: bool = False
        self._tracing_context = None

    def _sync_model_name_to_metrics(self):
        """Sync the current AI model name (in `self._settings.model`) for usage in metrics.

        We don't store model name here because there's already a single source
        of truth for it in `self._settings.model`. This method is just for
        syncing the model name to the metrics data.

        Args:
            model: The name of the AI model to use.
        """
        self.set_core_metrics_data(
            MetricsData(processor=self.name, model=self._settings.model or "")
        )

    async def start(self, frame: StartFrame):
        """Start the AI service.

        Called when the service should begin processing. Subclasses should
        override this method to perform service-specific initialization.

        Args:
            frame: The start frame containing initialization parameters.
        """
        self._settings.validate_complete()
        self._tracing_enabled = frame.enable_tracing
        self._tracing_context = frame.tracing_context

    async def stop(self, frame: EndFrame):
        """Stop the AI service.

        Called when the service should stop processing. Subclasses should
        override this method to perform cleanup operations.

        Args:
            frame: The end frame.
        """
        pass

    async def cancel(self, frame: CancelFrame):
        """Cancel the AI service.

        Called when the service should cancel all operations. Subclasses should
        override this method to handle cancellation logic.

        Args:
            frame: The cancel frame.
        """
        pass

    async def _update_settings(self, delta: ServiceSettings) -> Dict[str, Any]:
        """Apply a settings delta and return the changed fields.

        The delta is applied to ``_settings`` and a dict mapping each changed
        field name to its **pre-update** value is returned.  The ``model``
        field is handled specially: when it changes, ``set_model_name`` is
        called.

        Concrete services should override this method (calling ``super()``)
        to react to specific changed fields (e.g. reconnect on voice change).

        Args:
            delta: A delta-mode settings object.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = self._settings.apply_update(delta)

        if "model" in changed:
            self._sync_model_name_to_metrics()

        if changed:
            logger.info(f"{self.name}: updated settings fields: {set(changed)}")

        return changed

    def _warn_unhandled_updated_settings(self, unhandled):
        """Log a warning for settings changes that won't take effect at runtime.

        Convenience helper for ``_update_settings`` overrides.  Accepts any
        iterable of field names (a ``dict``, ``set``, ``dict_keys``, etc.).

        Args:
            unhandled: Field names that changed but are not applied.
        """
        if unhandled:
            fields = ", ".join(sorted(unhandled))
            logger.warning(f"{self.name}: runtime update of [{fields}] is not currently supported")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and handle service lifecycle.

        Automatically handles StartFrame, EndFrame, and CancelFrame by calling
        the appropriate lifecycle methods.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._start(frame)
        elif isinstance(frame, EndFrame):
            await self._stop(frame)
        elif isinstance(frame, CancelFrame):
            await self._cancel(frame)

    async def process_generator(self, generator: AsyncGenerator[Frame | None, None]):
        """Process frames from an async generator.

        Takes an async generator that yields frames and processes each one,
        handling error frames specially by pushing them as errors.

        Args:
            generator: An async generator that yields Frame objects or None.
        """
        async for f in generator:
            if f:
                if isinstance(f, ErrorFrame):
                    await self.push_error_frame(f)
                else:
                    await self.push_frame(f)

    async def _start(self, frame: StartFrame):
        try:
            await self.start(frame)
        except Exception as e:
            logger.error(f"{self}: exception processing {frame}: {e}")

    async def _stop(self, frame: EndFrame):
        try:
            await self.stop(frame)
        except Exception as e:
            logger.error(f"{self}: exception processing {frame}: {e}")

    async def _cancel(self, frame: CancelFrame):
        try:
            await self.cancel(frame)
        except Exception as e:
            logger.error(f"{self}: exception processing {frame}: {e}")
