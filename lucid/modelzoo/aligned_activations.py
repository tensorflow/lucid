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

from __future__ import absolute_import, division, print_function

from lucid.misc.io.sanitizing import sanitize
from lucid.misc.io import load

import numpy as np

PATH_TEMPLATE = "gs://modelzoo/aligned-activations/{}/{}-{:05d}-of-01000.npy"
PAGE_SIZE = 10000
NUMBER_OF_AVAILABLE_SAMPLES = 100000
assert NUMBER_OF_AVAILABLE_SAMPLES % PAGE_SIZE == 0
NUMBER_OF_PAGES = NUMBER_OF_AVAILABLE_SAMPLES // PAGE_SIZE


def get_aligned_activations(layer):
    activation_paths = [
        PATH_TEMPLATE.format(sanitize(layer.model_class.name), sanitize(layer.name), page)
        for page in range(NUMBER_OF_PAGES)
    ]
    activations = [load(path) for path in activation_paths]
    return np.vstack(activations)
