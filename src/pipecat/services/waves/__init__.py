import sys

from pipecat.services import DeprecatedModuleProxy

from .tts import *

sys.modules[__name__] = DeprecatedModuleProxy(globals(), "waves", "waves.tts")
