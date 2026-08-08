#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""VAD analyzer that reads an AICFilter's pre-enhancement predictions.

The ai-coustics SDK expects a VAD to run on the original signal, not on
enhancement output. In a Pipecat pipeline the enhancement filter is the only
component that sees audio before it is enhanced, so when both are used the VAD
lives inside :class:`pipecat.audio.filters.aic_filter.AICFilter` and this
analyzer reports what it detected.

Classes:
    AICFilterVADAnalyzer: Reads speech probability from an AICFilter's VAD.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams

if TYPE_CHECKING:
    from pipecat.audio.filters.aic_filter import AICFilter


class AICFilterVADAnalyzer(VADAnalyzer):
    """VAD analyzer backed by the VAD running inside an :class:`AICFilter`.

    The filter advances its VAD on each pre-enhancement block, so the
    probabilities this analyzer reports describe the original signal even though
    the analyzer itself sits downstream of enhancement. Its own input buffer is
    therefore ignored; it only paces how often the filter's latest prediction is
    read. The base :class:`VADAnalyzer` state machine gates speech start/stop
    from that probability using :class:`VADParams`.

    Use :class:`pipecat.audio.vad.aic_quail_vad.AICQuailVADAnalyzer` instead when
    no AIC enhancement is in the path — it owns its own VAD and needs no filter.

    Example::

        aic_filter = AICFilter(
            license_key=os.environ["AIC_SDK_LICENSE"],
            model_id="quail-vf-2.2-l-16khz",
            vad_model_id="vad-2.1-xxs-16khz",
        )
        vad_analyzer = AICFilterVADAnalyzer(aic_filter=aic_filter)
    """

    def __init__(
        self,
        *,
        aic_filter: AICFilter,
        sample_rate: int | None = None,
        params: VADParams | None = None,
    ) -> None:
        """Initialize the filter-bound VAD analyzer.

        Args:
            aic_filter: The filter whose VAD supplies predictions. It must have
                been constructed with ``vad_model_id`` or ``vad_model_path``.
            sample_rate: Initial sample rate; the pipeline sets this via
                :meth:`set_sample_rate` once the transport rate is known.
            params: Optional :class:`VADParams` for the base state machine.

        Raises:
            ValueError: If the filter has no VAD model configured.
        """
        if not aic_filter.has_vad_model:
            raise ValueError(
                "AICFilterVADAnalyzer requires an AICFilter with a VAD model. "
                "Pass 'vad_model_id' or 'vad_model_path' when constructing AICFilter, "
                "or use AICQuailVADAnalyzer for a standalone VAD."
            )

        super().__init__(sample_rate=sample_rate, params=params)

        self._filter = aic_filter
        # Latch so a not-yet-started filter is reported once rather than on
        # every window. Reset once the filter's VAD context becomes readable.
        self._not_ready_logged = False

    def num_frames_required(self) -> int:
        """Return the number of int16 frames per analysis window.

        Matches the filter's block size once it is known, so the pipeline reads
        the filter's VAD at roughly the rate the filter advances it.
        """
        frames = self._filter.frames_per_block
        if frames > 0:
            return frames
        # Pre-start fallback so the base class can compute internal sizes
        # before the filter has been given a sample rate.
        return int(self.sample_rate * 0.01) if self.sample_rate else 160

    def voice_confidence(self, buffer: bytes) -> float:
        """Read the filter's latest speech probability.

        Args:
            buffer: Ignored. The filter already ran the VAD on the original,
                pre-enhancement samples for this audio.

        Returns:
            The raw speech probability in ``[0.0, 1.0]``, or ``0.0`` while the
            filter has not started.
        """
        try:
            vad_ctx = self._filter.get_vad_context()
        except RuntimeError:
            if not self._not_ready_logged:
                logger.debug(
                    "AICFilterVADAnalyzer polled before AICFilter started; reporting no speech."
                )
                self._not_ready_logged = True
            return 0.0

        self._not_ready_logged = False

        try:
            probability = float(vad_ctx.raw_vad_probability())
        except Exception as e:  # noqa: BLE001 - keep the pipeline alive on SDK errors
            logger.error(f"AIC filter VAD read error: {e}")
            return 0.0

        # Clamp defensively to the [0.0, 1.0] the VADAnalyzer state machine expects.
        return max(0.0, min(1.0, probability))
