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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

from lucid.modelzoo.nets_factory import get_model, models_map
from lucid.modelzoo.vision_models import InceptionV1
from lucid.modelzoo.vision_base import Model, SerializedModel, FrozenGraphModel

def test_models_map():
  assert len(models_map) > 1
  assert Model.__name__ not in models_map
  assert SerializedModel.__name__ not in models_map
  assert FrozenGraphModel.__name__ not in models_map
  assert InceptionV1.__name__ in models_map

def test_get_model():
  model = get_model("InceptionV1")
  assert model is not None
  assert type(model) == InceptionV1

def test_get_model_fuzzy_feedback():
  with pytest.raises(ValueError) as excinfo:
    _ = get_model("InceptionV2")
  assert "InceptionV2_slim" in str(excinfo.value)
