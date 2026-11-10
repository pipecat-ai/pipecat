#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for warnings emitted when importing the TTS service module."""

import subprocess
import sys
import unittest

# Run in a fresh interpreter: the module may already be imported by other tests.
_IMPORT_AND_COLLECT = """
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import pipecat.services.tts_service
print("\\n".join(str(w.message) for w in caught if w.category is DeprecationWarning))
"""


class TestTTSServiceImportWarnings(unittest.TestCase):
    def test_import_does_not_warn_about_deprecated_tts_bases(self):
        result = subprocess.run(
            [sys.executable, "-c", _IMPORT_AND_COLLECT],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertNotIn("AudioContextTTSService", result.stdout)

    def test_audio_context_word_service_still_warns_when_subclassed(self):
        from pipecat.services.tts_service import AudioContextWordTTSService

        with self.assertWarns(DeprecationWarning):

            class _Service(AudioContextWordTTSService):
                pass


if __name__ == "__main__":
    unittest.main()
