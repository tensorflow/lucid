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
from cachetools.func import lru_cache

PATH_TEMPLATE = "gs://modelzoo/aligned-activations/{}/{}-{:05d}-of-01000.npy"
PAGE_SIZE = 10000
NUMBER_OF_AVAILABLE_SAMPLES = 100000
assert NUMBER_OF_AVAILABLE_SAMPLES % PAGE_SIZE == 0
NUMBER_OF_PAGES = NUMBER_OF_AVAILABLE_SAMPLES // PAGE_SIZE


@lru_cache()
def get_aligned_activations(layer):
    """Downloads 100k activations of the specified layer sampled from iterating over
    ImageNet. Activations of all layers where sampled at the same spatial positions for
    each image, allowing the calculation of correlations."""
    activation_paths = [
        PATH_TEMPLATE.format(
            sanitize(layer.model_class.name), sanitize(layer.name), page
        )
        for page in range(NUMBER_OF_PAGES)
    ]
    activations = [load(path) for path in activation_paths]
    return np.vstack(activations)


@lru_cache()
def layer_correlation(layer1, layer2=None):
    """Computes the correlation matrix between the neurons of two layers."""

    # if only one layer is passed, compute the correlation matrix of that layer
    if layer2 is None:
        layer2 = layer1

    act1, act2 = layer1.activations, layer2.activations
    covariance = np.matmul(act1.T, act2)
    return covariance / len(act1)


@lru_cache()
def layer_decorrelation(layer):
    """Inverse of a layer's correlation matrix. Function exists mostly for caching."""
    return np.linalg.inv(layer_correlation(layer))


def push_activations(activations, from_layer, to_layer):
    """Push activations from one model to another using prerecorded correlations"""
    decorrelation_matrix = layer_decorrelation(from_layer)
    activations_decorrelated = np.dot(decorrelation_matrix, activations.T).T
    correlation_matrix = layer_correlation(from_layer, to_layer)
    activation_recorrelated = np.dot(activations_decorrelated, correlation_matrix)
    return activation_recorrelated
