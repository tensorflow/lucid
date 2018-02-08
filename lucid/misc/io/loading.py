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

"""Methods for loading arbitrary data from arbitrary sources.

This module takes a URL, infers its underlying data type and how to locate it,
loads the data into memory and returns a convenient representation.

This should support for example PNG images, JSON files, npy files, etc.
"""

from __future__ import absolute_import, division, print_function

import os
import json
import logging
import numpy as np
import PIL.Image

from lucid.misc.io.reading import read_handle


# create logger with module name, e.g. lucid.misc.io.reading
log = logging.getLogger(__name__)


def _load_npy(handle, **kwargs):
  """Load npy file as numpy array."""
  del kwargs
  return np.load(handle)


def _load_img(handle, **kwargs):
  """Load image file as numpy array."""
  del kwargs
  # PIL.Image will infer image type from provided handle's file extension
  pil_img = PIL.Image.open(handle)
  img_array = np.asarray(pil_img)
  dtype = img_array.dtype
  # using np.divide should avoid an extra copy compared to doing division first
  return np.divide(img_array, np.iinfo(dtype).max, dtype=np.float32)


def _load_json(handle, **kwargs):
  """Load json file as python object."""
  del kwargs
  return json.load(handle)


def _load_text(handle, encoding='utf-8'):
  """Load and decode a string."""
  return handle.read().decode(encoding)

loaders = {
  ".png":  _load_img,
  ".jpg":  _load_img,
  ".jpeg": _load_img,
  ".npy":  _load_npy,
  ".npz":  _load_npy,
  ".json": _load_json,
  ".txt": _load_text,
  ".md": _load_text,
}


def load(url, cache=None, encoding='utf-8'):
  """Load a file.

  File format is inferred from url. File retrieval strategy is inferred from
  URL. Returned object type is inferred from url extension.

  Args:
    path: a (reachable) URL

  Raises:
    RuntimeError: If file extension or URL is not supported.
  """
  _, ext = os.path.splitext(url)
  if not ext:
    raise RuntimeError("No extension in URL: " + url)

  ext = ext.lower()
  if ext in loaders:
    loader = loaders[ext]
    message = "Using inferred loader '%s' due to passed file extension '%s'."
    log.debug(message, loader.__name__[6:], ext)
    with read_handle(url, cache=cache) as handle:
      result = loader(handle, encoding=encoding)
    return result
  else:
    log.warn("Unknown extension '%s', attempting to load as image.", ext)
    try:
      with read_handle(url, cache=cache) as handle:
        result = _load_img(handle)
    except Exception as e:
      message = "Could not load resource %s as image. Supported extensions: %s"
      log.error(message, url, list(loaders))
      raise RuntimeError(message.format(url, list(loaders)))
    else:
      log.info("Unknown extension '%s' successfully loaded as image.", ext)
      return result
