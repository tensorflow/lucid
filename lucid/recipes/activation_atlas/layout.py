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

import numpy as np
import logging
from umap import UMAP

log = logging.getLogger(__name__)


def normalize_layout(layout, min_percentile=1, max_percentile=99, relative_margin=0.1):
    """Removes outliers and scales layout to between [0,1]."""

    # compute percentiles
    mins = np.percentile(layout, min_percentile, axis=(0))
    maxs = np.percentile(layout, max_percentile, axis=(0))

    # add margins
    mins -= relative_margin * (maxs - mins)
    maxs += relative_margin * (maxs - mins)

    # `clip` broadcasts, `[None]`s added only for readability
    clipped = np.clip(layout, mins, maxs)

    # embed within [0,1] along both axes
    clipped -= clipped.min(axis=0)
    clipped /= clipped.max(axis=0)

    return clipped


def aligned_umap(activations, umap_options={}, normalize=True, verbose=False):
    """`activations` can be a list of ndarrays. In that case a list of layouts is returned."""

    umap_defaults = dict(
        n_components=2, n_neighbors=50, min_dist=0.05, verbose=verbose, metric="cosine"
    )
    umap_defaults.update(umap_options)

    # if passed a list of activations, we combine them and later split the layouts
    if type(activations) is list or type(activations) is tuple:
        num_activation_groups = len(activations)
        combined_activations = np.concatenate(activations)
    else:
        num_activation_groups = 1
        combined_activations = activations
    try:
        layout = UMAP(**umap_defaults).fit_transform(combined_activations)
    except (RecursionError, SystemError) as exception:
        log.error("UMAP failed to fit these activations. We're not yet sure why this sometimes occurs.")
        raise ValueError("UMAP failed to fit activations: %s", exception)

    if normalize:
        layout = normalize_layout(layout)

    if num_activation_groups > 1:
        layouts = np.split(layout, num_activation_groups, axis=0)
        return layouts
    else:
        return layout
