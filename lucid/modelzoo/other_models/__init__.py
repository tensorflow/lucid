from lucid.modelzoo.vision_base import Model
from lucid.modelzoo.other_models.InceptionV1 import InceptionV1

__all__ = [obj for obj in globals().values()
           if isinstance(obj, type) and issubclass(obj, Model) 
           and obj is not Model ]
