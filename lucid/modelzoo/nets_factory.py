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

"""Contains a factory function for accessing models.

You can either use the provided `get_model` function, or directly access the
`models_map` variable containing a dictionary from a model name to its class.
"""

from __future__ import absolute_import, division, print_function

import inspect
import tensorflow as tf

from lucid.modelzoo import vision_models
from lucid.modelzoo import vision_base


def _generate_models_map():
    base_classes = inspect.getmembers(vision_base, inspect.isclass)

    list_all_models = []
    list_all_models += inspect.getmembers(vision_models, inspect.isclass)

    list_filtered = filter(lambda c: c not in base_classes, list_all_models)
    return dict(list_filtered)


models_map = _generate_models_map()


def get_model(name):
    """Returns a model instance such as `model = vision_models.InceptionV1()`.
    In the future may be expanded to filter by additional criteria, such as
    architecture, dataset, and task the model was trained on.
    Args:
      name: The name of the model, as given by the class name in vision_models.
    Returns:
      An instantiated Model class with the requested model. Users still need to
      manually `load_graphdef` on the return value, and manually import this
      model's graph into their current graph.
    Raises:
      ValueError: If network `name` is not recognized.
    """
    if name not in models_map:
        candidates = filter(lambda key: name in key, models_map.keys())
        candidates_string = ", ".join(candidates)
        raise ValueError(
            "No network named {}. Did you mean one of {}?".format(
                name, candidates_string
            )
        )

    model_class = models_map[name]
    model = model_class()
    return model
