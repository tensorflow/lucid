from lucid.modelzoo.vision_base import Model

from lucid.modelzoo.slim_models.Inception import *
from lucid.modelzoo.slim_models.ResNetV1 import *
from lucid.modelzoo.slim_models.ResNetV2 import *
from lucid.modelzoo.slim_models.others import *


__all__ = [name for name, obj in list(globals().items())
           if isinstance(obj, type) and issubclass(obj, Model)
           and obj is not Model]
