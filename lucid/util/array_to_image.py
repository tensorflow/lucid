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
  logging.info(message.format(image_mode, rank, shape))
  return image_mode


def _infer_domain_from_array(array):
  """Guesses a canonical domain from an arrays domain.
  Covers common cases such as 0,1, -1,1, 0,255.

  Args:
    array: a numpy array

  Returns:
    inferred canonical domain of the array as a tuple (low, high)
  """
  low, high = np.min(array), np.max(array)
  assert low <= high

  try:
    if low >= 0:
      if high <= 1:
        domain = (0, 1)
      elif high <= 255:
        domain = (0, 255)
      else:
        raise CouldNotInfer
    elif low >= -1:
      if high <= 0:
        domain = (-1, 0)
      elif high <= 1:
        domain = (-1, 1)
      else:
        raise CouldNotInfer
    else:
      raise CouldNotInfer
  except CouldNotInfer:
    message = "Could not infer canonical domain from (~{:.2f}, ~{:.2f})"
    logging.warn(message.format(low, high))
    domain = (low, high)
  else:
    message = "Inferred canonical domain {} from (~{:.2f}, ~{:.2f})"
    logging.info(message.format(domain, low, high))
  finally:
    assert domain is not None
  return domain


def _normalize_array_and_convert_to_image(array, domain=None, w=None):
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

  domain = domain or _infer_domain_from_array(array)

  low, high = np.min(array), np.max(array)
  if low < domain[0] or high > domain[1]:
    message = "Clipping domain from (~{:.2f}, ~{:.2f}) to (~{:.2f}, ~{:.2f})"
    logging.info(message.format(low, high, domain[0], domain[1]))
    array = array.clip(*domain)

  force_stretching_to_domain = False
  if np.issubdtype(array.dtype, np.signedinteger):
    if low >= 0:
      array = np.uint8(array)
    else:
      force_stretching_to_domain = True

  if np.issubdtype(array.dtype, np.inexact) or force_stretching_to_domain:
    message = "Stretching domain from (~{:.2f}, ~{:.2f}) to (0, 255)."
    logging.info(message.format(low, high))
    divisor = domain[1] - domain[0]
    offset = domain[0]
    array = 255 * (array - offset) / divisor
    array = np.uint8(array)

  assert np.issubdtype(array.dtype, np.unsignedinteger)

  image_mode = _infer_image_mode_from_shape(array.shape)
  image = PIL.Image.fromarray(array, mode=image_mode)

  if w is not None:
    # TODO: is that intended? feels like it should just be shape[0]
    original_w = min(array.shape[0], array.shape[1])
    if original_w != w:
      aspect = float(image.size[0]) / image.size[1]
      h = int(w / aspect)
      image = image.resize((w, h), PIL.Image.NEAREST)

  return image
