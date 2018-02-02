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

"""Methods for loading arbitrary data from arbitrary sources.

This module takes a URL, infers its underlying data type and how to locate it,
loads the data into memory and returns a convenient representation.

This should support for example PNG images, JSON files, npy files, etc.
"""

from __future__ import absolute_import, division, print_function

import os
import json
import numpy as np
import PIL.Image

from lucid.util.read import reading


def load_npy(handle):
  """Load npy file as numpy array."""
  return np.load(handle)


def load_img(handle):
  """Load image file as numpy array."""
  # PIL requires a buffer interface, so wrap data in BytesIO
  pil_img = PIL.Image.open(handle)
  return np.asarray(pil_img)


def load_json(handle):
  """Load json file as python object."""
  return json.load(handle)


loaders = {
  ".png": load_img,
  ".jpg": load_img,
  ".jpeg": load_img,
  ".npy": load_npy,
  ".npz": load_npy,
  ".json": load_json,
}


def load(url):
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

  if ext in loaders:
    loader = loaders[ext]
    with reading(url) as handle:
      result = loader(handle)
    return result
  else:
    message = "Unknown extension '{}', supports {}."
    raise RuntimeError(message.format(ext, loaders))
