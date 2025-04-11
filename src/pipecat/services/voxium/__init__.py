import sys

from pipecat.services import DeprecatedModuleProxy

from .stt import *

sys.modules[__name__] = DeprecatedModuleProxy(globals(), "voxium", "voxium.stt")
