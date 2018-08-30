# Copyright 2018 The Lucid Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Clean export of vision_models.

We manually remove the following symbols from this module to keep tab
completion as clean as possible--even when it doesn't respect `__all__`.
Clean namespaces for those lucid.modelzoo modules that contain models are
enforced by tests in test/modelzoo/test_vision_models.
"""

from lucid.modelzoo.vision_base import Model, Layer

from lucid.modelzoo.caffe_models import *
from lucid.modelzoo.slim_models import *
from lucid.modelzoo.other_models import *


__all__ = [_name for _name, _obj in list(globals().items())
           if isinstance(_obj, type) and issubclass(_obj, Model)]

# in Python 2 only, list comprehensions leak bound vars to a broader scope
try:
  del _obj
  del _name
except:
  pass
