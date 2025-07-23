import sys

from pipecat.services import DeprecatedModuleProxy

from .llm import *
from .stt import *
from .tts import *

sys.modules[__name__] = DeprecatedModuleProxy(globals(), "baseten", "baseten.[llm,stt,tts]")