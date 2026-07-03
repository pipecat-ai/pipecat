#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Verify ``AICVADAnalyzer`` emits a DeprecationWarning on construction."""

import unittest
import warnings

try:
    import aic_sdk  # noqa: F401

    HAS_AIC_SDK = True
except ImportError:
    HAS_AIC_SDK = False


@unittest.skipUnless(HAS_AIC_SDK, "aic-sdk not installed")
class TestAICVADAnalyzerDeprecation(unittest.TestCase):
    def test_construction_emits_deprecation_warning(self):
        from pipecat.audio.vad.aic_vad import AICVADAnalyzer

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            AICVADAnalyzer()

        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(len(deprecations), 1)
        message = str(deprecations[0].message)
        self.assertIn("`AICVADAnalyzer` is deprecated", message)
        self.assertIn("AICQuailVADAnalyzer", message)
        self.assertIn("1.6.0", message)


if __name__ == "__main__":
    unittest.main()
