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

from enum import Enum, auto

import numpy as np

from lucid.modelzoo.aligned_activations import (
    push_activations,
    NUMBER_OF_AVAILABLE_SAMPLES,
    layer_inverse_covariance,
)
from lucid.recipes.activation_atlas.layout import aligned_umap
from lucid.recipes.activation_atlas.render import render_icons
from more_itertools import chunked


def activation_atlas(
    model,
    layer,
    grid_size=10,
    icon_size=96,
    number_activations=NUMBER_OF_AVAILABLE_SAMPLES,
    icon_batch_size=32,
    verbose=False,
):
    """Renders an Activation Atlas of the given model's layer."""

    activations = layer.activations[:number_activations, ...]
    layout, = aligned_umap(activations, verbose=verbose)
    directions, coordinates, _ = bin_laid_out_activations(
        layout, activations, grid_size
    )
    icons = []
    for directions_batch in chunked(directions, icon_batch_size):
        icon_batch, losses = render_icons(
            directions_batch, model, layer=layer.name, size=icon_size, num_attempts=1
        )
        icons += icon_batch
    canvas = make_canvas(icons, coordinates, grid_size)

    return canvas


def aligned_activation_atlas(
    model1,
    layer1,
    model2,
    layer2,
    grid_size=10,
    icon_size=80,
    num_steps=1024,
    whiten_layers=True,
    number_activations=NUMBER_OF_AVAILABLE_SAMPLES,
    icon_batch_size=32,
    verbose=False,
):
    """Renders two aligned Activation Atlases of the given models' layers.

    Returns a generator of the two atlasses, and a nested generator for intermediate
    atlasses while they're being rendered.
    """
    combined_activations = _combine_activations(
        layer1, layer2, number_activations=number_activations
    )
    layouts = aligned_umap(combined_activations, verbose=verbose)

    for model, layer, layout in zip((model1, model2), (layer1, layer2), layouts):
        directions, coordinates, densities = bin_laid_out_activations(
            layout, layer.activations[:number_activations, ...], grid_size, threshold=10
        )

        def _progressive_canvas_iterator():
            icons = []
            for directions_batch in chunked(directions, icon_batch_size):
                icon_batch, losses = render_icons(
                    directions_batch,
                    model,
                    alpha=False,
                    layer=layer.name,
                    size=icon_size,
                    n_steps=num_steps,
                    S=layer_inverse_covariance(layer) if whiten_layers else None,
                )
                icons += icon_batch
                yield make_canvas(icons, coordinates, grid_size)

        yield _progressive_canvas_iterator()


# Helpers


class ActivationTranslation(Enum):
    ONE_TO_TWO = auto()
    BIDIRECTIONAL = auto()


def _combine_activations(
    layer1,
    layer2,
    activations1=None,
    activations2=None,
    mode=ActivationTranslation.BIDIRECTIONAL,
    number_activations=NUMBER_OF_AVAILABLE_SAMPLES,
):
    """Given two layers, combines their activations according to mode.

    ActivationTranslation.ONE_TO_TWO:
      Translate activations of layer1 into the space of layer2, and return a tuple of
      the translated activations and the original layer2 activations.

    ActivationTranslation.BIDIRECTIONAL:
      Translate activations of layer1 into the space of layer2, activations of layer2
      into the space of layer 1, concatenate them along their channels, and returns a
      tuple of the concatenated activations for each layer.
    """
    activations1 = activations1 or layer1.activations[:number_activations, ...]
    activations2 = activations2 or layer2.activations[:number_activations, ...]

    if mode is ActivationTranslation.ONE_TO_TWO:

        acts_1_to_2 = push_activations(activations1, layer1, layer2)
        return acts_1_to_2, activations2

    elif mode is ActivationTranslation.BIDIRECTIONAL:

        acts_1_to_2 = push_activations(activations1, layer1, layer2)
        acts_2_to_1 = push_activations(activations2, layer2, layer1)

        activations_model1 = np.concatenate((activations1, acts_1_to_2), axis=1)
        activations_model2 = np.concatenate((acts_2_to_1, activations2), axis=1)

        return activations_model1, activations_model2


def bin_laid_out_activations(layout, activations, grid_size, threshold=5):
    """Given a layout and activations, overlays a grid on the layout and returns
    averaged activations for each grid cell. If a cell contains less than `threshold`
    activations it will be discarded, so the number of returned data is variable."""

    assert layout.shape[0] == activations.shape[0]

    # calculate which grid cells each activation's layout position falls into
    # first bin stays empty because nothing should be < 0, so we add an extra bin
    bins = np.linspace(0, 1, num=grid_size + 1)
    bins[-1] = np.inf  # last bin should include all higher values
    indices = np.digitize(layout, bins) - 1  # subtract 1 to account for empty first bin

    # because of thresholding we may need to return a variable number of means
    means, coordinates, counts = [], [], []

    # iterate over all grid cell coordinates to compute their average directions
    grid_coordinates = np.indices((grid_size, grid_size)).transpose().reshape(-1, 2)
    for xy_coordinates in grid_coordinates:
        mask = np.equal(xy_coordinates, indices).all(axis=1)
        count = np.count_nonzero(mask)
        if count > threshold:
            counts.append(count)
            coordinates.append(xy_coordinates)
            mean = np.average(activations[mask], axis=0)
            means.append(mean)

    assert len(means) == len(coordinates) == len(counts)
    if len(coordinates) == 0:
        raise RuntimeError("Binning activations led to 0 cells containing activations!")

    return means, coordinates, counts


def make_canvas(icon_batch, coordinates, grid_size):
    """Given a list of images and their coordinates, places them on a white canvas."""

    grid_shape = (grid_size, grid_size)
    icon_shape = icon_batch[0].shape
    canvas = np.ones((*grid_shape, *icon_shape))

    for icon, (x, y) in zip(icon_batch, coordinates):
        canvas[x, y] = icon

    return np.hstack(np.hstack(canvas))


if __name__ == "__main__":
    activation_atlas()
