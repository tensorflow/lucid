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

import numpy as np
from enum import Enum, auto

from lucid.modelzoo.aligned_activations import (
    push_activations,
    NUMBER_OF_AVAILABLE_SAMPLES,
)
from lucid.recipes.activation_atlas.layout import aligned_umap
from lucid.recipes.activation_atlas.render import render_icons
from lucid.misc.batching import batch


def activation_atlas(
    model,
    layer,
    grid_size=10,
    icon_size=96,
    number_activations=NUMBER_OF_AVAILABLE_SAMPLES,
    verbose=False,
):
    """Renders an Activation Atlas of the given model's layer."""

    activations = layer.activations[:number_activations, ...]
    layout, = aligned_umap(activations, verbose=verbose)
    directions, coordinates, _ = _bin_laid_out_activations(
        layout, activations, grid_size
    )
    icons = []
    for directions_batch in batch(directions, batch_size=64):
        icon_batch, losses = render_icons(
            directions_batch, model, layer=layer.name, size=icon_size, num_attempts=1
        )
        icons += icon_batch
    canvas = _make_canvas(icons, coordinates, grid_size)

    return canvas


def aligned_activation_atlas(
    model1,
    layer1,
    model2,
    layer2,
    grid_size=10,
    icon_size=96,
    number_activations=NUMBER_OF_AVAILABLE_SAMPLES,
    verbose=False,
):
    combined_activations = _combine_activations(
        layer1, layer2, number_activations=number_activations
    )
    layouts = aligned_umap(combined_activations, verbose=verbose)

    atlasses = []
    for model, layer, layout in zip((model1, model2), (layer1, layer2), layouts):
        directions, coordinates, densities = _bin_laid_out_activations(
            layout, layer.activations[:number_activations, ...], grid_size
        )
        icons = []
        for directions_batch in batch(directions, batch_size=64):
            icon_batch, losses = render_icons(
                directions_batch,
                model,
                alpha=False,
                layer=layer.name,
                size=icon_size,
                num_attempts=1,
                n_steps=1024,
            )
            icons += icon_batch
        canvas = _make_canvas(icons, coordinates, grid_size)
        atlasses.append(canvas)

    return atlasses


# Helpers


class ActivationTranslation(Enum):
    ONE_TO_TWO = auto()
    BIDIRECTIONAL = auto()


def _combine_activations(
    layer1,
    layer2,
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
    activations1 = layer1.activations[:number_activations, ...]
    activations2 = layer2.activations[:number_activations, ...]

    if mode is ActivationTranslation.ONE_TO_TWO:

        acts_1_to_2 = push_activations(activations1, layer1, layer2)
        return acts_1_to_2, activations2

    elif mode is ActivationTranslation.BIDIRECTIONAL:

        acts_1_to_2 = push_activations(activations1, layer1, layer2)
        acts_2_to_1 = push_activations(activations2, layer2, layer1)

        activations_model1 = np.concatenate((activations1, acts_1_to_2), axis=1)
        activations_model2 = np.concatenate((acts_2_to_1, activations2), axis=1)

        return activations_model1, activations_model2


def _bin_laid_out_activations(layout, activations, grid_size, threshold=5):
    """Given a layout and activations, overlays a grid on the layout and returns
    averaged activations for each grid cell. If a cell contains less than `threshold`
    activations it will not be used, so the number of returned directions is variable."""

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
    for xy in grid_coordinates:
        mask = np.equal(xy, indices).all(axis=1)
        count = np.count_nonzero(mask)
        if count > threshold:
            counts.append(count)
            coordinates.append(xy)
            mean = np.average(activations[mask], axis=0)
            means.append(mean)

    assert len(means) == len(coordinates) == len(counts)

    return np.array(means), np.array(coordinates), np.array(counts)


def _make_canvas(icon_batch, coordinates, grid_size):
    """Given a list of images and their coordinates, places them on a white canvas."""

    grid_shape = (grid_size, grid_size)
    icon_shape = icon_batch[0].shape
    canvas = np.ones((*grid_shape, *icon_shape))

    for (x, y), icon in zip(coordinates, icon_batch):
        canvas[x, y] = icon

    return np.hstack(np.hstack(canvas))


if __name__ == "__main__":
    activation_atlas()
