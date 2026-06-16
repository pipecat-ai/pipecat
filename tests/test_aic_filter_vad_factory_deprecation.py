#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Verify ``AICFilter.create_vad_analyzer`` emits a DeprecationWarning."""

import unittest
import warnings

try:
    import aic_sdk  # noqa: F401

    HAS_AIC_SDK = True
except ImportError:
    HAS_AIC_SDK = False


@unittest.skipUnless(HAS_AIC_SDK, "aic-sdk not installed")
class TestAICFilterCreateVADAnalyzerDeprecation(unittest.TestCase):
    def test_factory_emits_deprecation_warning(self):
        from pipecat.audio.filters.aic_filter import AICFilter

        # Bypass __init__ so we don't need a real license/processor.
        filter_instance = AICFilter.__new__(AICFilter)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            filter_instance.create_vad_analyzer()

        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        # Exactly one warning expected: the factory's own. The inner
        # AICVADAnalyzer DeprecationWarning is suppressed by the factory's
        # catch_warnings(filterwarnings("ignore", ...)) block. Asserting
        # equality (not >= 1) so a regression that lets the inner warning
        # leak fails this test.
        self.assertEqual(len(deprecations), 1)
        self.assertIn("create_vad_analyzer", str(deprecations[0].message))
        self.assertIn("AICQuailVADAnalyzer", str(deprecations[0].message))
        self.assertIn("1.6.0", str(deprecations[0].message))


if __name__ == "__main__":
    unittest.main()
