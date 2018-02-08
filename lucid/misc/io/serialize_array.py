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

"""Utilities for normalizing arrays and converting them to images."""

from __future__ import absolute_import, division, print_function

import base64
import logging
import numpy as np
import PIL.Image
from io import BytesIO


# create logger with module name, e.g. lucid.misc.io.array_to_image
log = logging.getLogger(__name__)


def _normalize_array(array, domain=(0, 1)):
  """Given an arbitrary rank-3 NumPy array, produce one representing an image.

  This ensures the resulting array has a dtype of uint8 and a domain of 0-255.

  Args:
    array: NumPy array representing the image
    domain: expected range of values in array,
      defaults to (0, 1), if explicitly set to None will use the array's
      own range of values and normalize them.

  Returns:
    normalized PIL.Image
  """
  # first copy the input so we're never mutating the user's data
  array = np.array(array)
  # squeeze helps both with batch=1 and B/W and PIL's mode inference
  array = np.squeeze(array)
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


def _serialize_normalized_array(array, fmt='png', quality=70):
  """Given a normalized array, returns byte representation of image encoding.

  Args:
    array: NumPy array of dtype uint8 and range 0 to 255
    fmt: string describing desired file format, defaults to 'png'
    quality: specifies compression quality from 0 to 100 for lossy formats

  Returns:
    image data as BytesIO buffer
  """
  dtype = array.dtype
  assert np.issubdtype(dtype, np.unsignedinteger)
  assert np.max(array) <= np.iinfo(dtype).max
  assert array.shape[-1] > 1  # array dims must have been squeezed

  image = PIL.Image.fromarray(array)
  image_bytes = BytesIO()
  image.save(image_bytes, fmt, quality=quality)
  # TODO: Python 3 could save a copy here by using `getbuffer()` instead.
  image_data = image_bytes.getvalue()
  return image_data


def serialize_array(array, domain=(0, 1), fmt='png', quality=70):
  """Given an arbitrary rank-3 NumPy array,
  returns the byte representation of the encoded image.

  Args:
    array: NumPy array of dtype uint8 and range 0 to 255
    domain: expected range of values in array, see `_normalize_array()`
    fmt: string describing desired file format, defaults to 'png'
    quality: specifies compression quality from 0 to 100 for lossy formats

  Returns:
    image data as BytesIO buffer
  """
  normalized = _normalize_array(array, domain=domain)
  return _serialize_normalized_array(normalized, fmt=fmt, quality=quality)


JS_ARRAY_TYPES = {
    'int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32', 'float32', 'float64'
}


def array_to_jsbuffer(array):
  """Serialize 1d NumPy array to JS TypedArray.

  Data is serialized to base64-encoded string, which is much faster
  and memory-efficient than json list serialization.

  Args:
    array: 1d NumPy array, dtype must be one of JS_ARRAY_TYPES.

  Returns:
    JS code that evaluates to a TypedArray as string.

  Raises:
    TypeError: if array dtype or shape not supported.
  """
  if array.ndim != 1:
    raise TypeError('Only 1d arrays can be converted JS TypedArray.')
  if array.dtype.name not in JS_ARRAY_TYPES:
    raise TypeError('Array dtype not supported by JS TypedArray.')
  js_type_name = array.dtype.name.capitalize() + 'Array'
  data_base64 = base64.b64encode(array.tobytes()).decode('ascii')
  code = """
    (function() {
      const data = atob("%s");
      const buf = new Uint8Array(data.length);
      for (var i=0; i<data.length; ++i) {
        buf[i] = data.charCodeAt(i);
      }
      var array_type = %s;
      if (array_type == Uint8Array) {
        return buf;
      }
      return new array_type(buf.buffer);
    })()
  """ % (data_base64, js_type_name)
  return code
