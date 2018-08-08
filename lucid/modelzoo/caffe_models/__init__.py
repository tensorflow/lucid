"""Clean export of caffe_models.

We manually remove the following symbols from this module to keep tab
completion as clean as possible--even when it doesn't respect `__all__`.
Clean namespaces for those lucid.modelzoo modules that contain models are
enforced by tests in test/modelzoo/test_vision_models.
"""

from lucid.modelzoo.vision_base import Model as _Model

from lucid.modelzoo.caffe_models.AlexNet import *
from lucid.modelzoo.caffe_models.InceptionV1 import *
from lucid.modelzoo.caffe_models.others import *

__all__ = [_name for _name, _obj in list(globals().items())
           if isinstance(_obj, type) and issubclass(_obj, _Model)
           and _obj is not _Model]


del absolute_import
del division
del print_function

del IMAGENET_MEAN_BGR

# in Python 2 only, list comprehensions leak bound vars to a broader scope
try:
  del _obj
  del _name
except:
  pass
