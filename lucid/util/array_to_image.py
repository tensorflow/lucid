# Copyright 2018 The Deepviz Authors. All Rights Reserved.
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

"""Utilities for normalizing arrays and converting them to images."""

from __future__ import absolute_import, division, print_function

import logging
import numpy as np
import PIL.Image
from io import BytesIO


# create logger with module name, e.g. lucid.util.array_to_image
log = logging.getLogger(__name__)


class CouldNotInfer(Exception):
  pass


def _infer_image_mode_from_shape(shape):
  """Guesses image mode from supplied shape.

  For example assumes that a 2D array is meant as a B/W image.

  Args:
    shape: an enumerable representation of dimensions, the length of which is
      assumed to be its rank. E.g. (200,200,3) for a 200 by 200 pixel RGB image
  Returns:
    image_mode: a string representing grayscale, RGB, or RGBA image modes.
  """
  rank = len(shape)

  if rank == 2:
    image_mode = "L"
  elif rank == 3:
    depth = shape[-1]
    if depth == 1:
      image_mode = "L"
    elif depth == 3:
      image_mode = "RGB"
    elif depth == 4:
      image_mode = "RGBA"
    else:
      raise ValueError("""Can only infer image mode of 2D, 3D & 4D images,
                         invalid shape: """ + str(shape))
  elif rank == 4:
    raise ValueError("""Can not infer image mode for shape of rank 4.
      You may be trying to infer from an image with batch dimension?""")

  message = "Inferred image mode '{}' from rank-{:d} shape {}"
  log.info(message.format(image_mode, rank, shape))
  return image_mode


def _normalize_array(array, domain=(0, 1)):
  """Normalize an image pixel values and width.

  Args:
    a: NumPy array representing the image
    domain: Domain of pixel values, inferred from min & max values if None
    w: width of output image, scaled using nearest neighbor interpolation.
      size unchanged if None

  Returns:
    normalized PIL.Image
  """
  # squeeze helps both with batch=1 and B/W
  array = np.squeeze(np.asarray(array))
  assert len(array.shape) <= 3
  assert np.issubdtype(array.dtype, np.number)

  low, high = np.min(array), np.max(array)
  if domain is None:
    message = "No domain specified, normalizing from measured (~%.2f, ~%.2f)"
    log.debug(message, low, high)
    domain = (low, high)

  # clip values if domain was specified and array contains values outside of it
  if low < domain[0] or high > domain[1]:
    message = "Clipping domain from (~{:.2f}, ~{:.2f}) to (~{:.2f}, ~{:.2f})."
    log.info(message.format(low, high, domain[0], domain[1]))
    array = array.clip(*domain)

  min_value, max_value = 0, np.iinfo(np.uint8).max  # = 255
  # convert signed to unsigned if needed
  if np.issubdtype(array.dtype, np.inexact):
    offset = domain[0]
    if offset != 0:
      array -= offset
      log.debug("Converting inexact array by subtracting -%.2f.", offset)
    scalar = max_value / (domain[1] - domain[0])
    if scalar != 1:
      array *= scalar
      log.debug("Converting inexact array by scaling by %.2f.", scalar)

  assert np.max(array) <= max_value and np.min(array) >= min_value
  array = array.astype(np.uint8)

  return array


def _serialize_array(array, fmt='png', quality=70):
  assert np.issubdtype(array.dtype, np.unsignedinteger)
  assert np.max(array) <= 255
  inferred_mode = _infer_image_mode_from_shape(array.shape)
  image = PIL.Image.fromarray(array, mode=inferred_mode)
  image_bytes = BytesIO()
  image.save(image_bytes, fmt, quality=quality)
  # TODO: Python 3 could save a copy here by using `getbuffer()` instead.
  image_data = image_bytes.getvalue()
  return image_data
