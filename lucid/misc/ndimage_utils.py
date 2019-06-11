# Copyright 2019 The Lucid Authors. All Rights Reserved.
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
from scipy import ndimage


def resize(image, target_size=None, ratios=None, **kwargs):
    """Resize an ndarray image of rank 3 or 4.
    target_size can be a tuple `(width, height)` or scalar `width`.
    Alternatively you can directly specify the ratios by which each
    dimension should be scaled, or a single ratio"""

    # input validation
    if target_size is None:
      assert ratios is not None
    else:
      if isinstance(target_size, int):
          target_size = (target_size, target_size)

      if not isinstance(target_size, (list, tuple, np.ndarray)):
          message = (
              "`target_size` should be a single number (width) or a list"
              "/tuple/ndarray (width, height), not {}.".format(type(target_size))
          )
          raise ValueError(message)

    rank = len(image.shape)
    assert 3 <= rank <= 4

    original_size = image.shape[-3:-1]


    ratios_are_noop = all(ratio == 1 for ratio in ratios) if ratios is not None else False
    target_size_is_noop = target_size == original_size if target_size is not None else False
    if ratios_are_noop or target_size_is_noop:
        return image  # noop return because ndimage.zoom doesn't check itself

    # TODO: maybe allow -1 in target_size to signify aspect-ratio preserving resize?
    ratios = ratios or [t / o for t, o in zip(target_size, original_size)]
    zoom = [1] * rank
    zoom[-3:-1] = ratios

    roughly_resized = ndimage.zoom(image, zoom, **kwargs)
    if target_size is not None:
      return roughly_resized[..., : target_size[0], : target_size[1], :]
    else:
      return roughly_resized


def composite(
    background_image,
    foreground_image,
    foreground_width_ratio=0.25,
    foreground_position=(0.0, 0.0),
):
    """Takes two images and composites them."""

    if foreground_width_ratio <= 0:
        return background_image

    composite = background_image.copy()
    width = int(foreground_width_ratio * background_image.shape[1])
    foreground_resized = resize(foreground_image, width)
    size = foreground_resized.shape

    x = int(foreground_position[1] * (background_image.shape[1] - size[1]))
    y = int(foreground_position[0] * (background_image.shape[0] - size[0]))

    # TODO: warn if resulting coordinates are out of bounds?
    composite[y : y + size[0], x : x + size[1]] = foreground_resized

    return composite


def soft_alpha_blend(image_with_alpha, amount_background=.333, gamma=2.2):
    assert image_with_alpha.shape[-1] == 4

    alpha = image_with_alpha[..., -1][..., np.newaxis]
    rgb = image_with_alpha[..., :3]
    white = np.ones_like(rgb)
    background = amount_background * rgb + (1 - amount_background) * white
    blended = np.power(
        (alpha * np.power(rgb, gamma) + (1 - alpha) * np.power(background, gamma)),
        1 / gamma,
    )
    return blended

