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

"""Method for saving arbitrary data to arbitrary destinations.

This module takes an object and URL, infers how to serialize and how to write
it out to the destination.

If an object could have multiple representations, this tries to infer the
intended representation from the URL's file extension.

Possible extension: if not given a URL this could create one and return it?
"""

from __future__ import absolute_import, division, print_function

import logging
import os.path
import json
import numpy as np
import PIL.Image

from lucid.misc.io.writing import write, write_handle
from lucid.misc.io.serialize_array import _normalize_array


# create logger with module name, e.g. lucid.misc.io.saving
log = logging.getLogger(__name__)


def save_json(object, url, indent=2):
  """Save object as json on CNS."""
  obj_json = json.dumps(object, indent=indent)
  write(obj_json, url, 'w')


def save_npy(object, url):
  """Save numpy array as npy file."""
  with write_handle(url, "w") as handle:
    np.save(handle, object)


def save_npz(object, url):
  """Save dict of numpy array as npz file."""
  if type(object) is dict:
    np.savez(url, **object)
  elif type(object) is list:
    np.savez(url, *object)
  else:
    log.warn("Saving non dict or list as npz file, did you maybe want npy?")
    np.savez(url, object)


def save_img(object, url, **kwargs):
  """Save numpy array as image file on CNS."""
  if isinstance(object, np.ndarray):
    normalized = _normalize_array(object)
    image = PIL.Image.fromarray(normalized)
  elif not isinstance(object, PIL.Image):
    raise ValueError("Can only save_img for numpy arrays or PIL.Images!")

  with write_handle(url) as handle:
    image.save(handle, **kwargs)  # will infer format from handle's url ext.


savers = {
  ".png": save_img,
  ".jpg": save_img,
  ".jpeg": save_img,
  ".npy": save_npy,
  ".npz": save_npz,
  ".json": save_json,
}


def save(thing, url, **kwargs):
  """Save object to file on CNS.

  File format is inferred from path. Use save_img(), save_npy(), or save_json()
  if you need to force a particular format.

  Args:
    obj: object to save.
    path: CNS path.

  Raises:
    RuntimeError: If file extension not supported.
  """
  _, ext = os.path.splitext(url)
  if not ext:
    raise RuntimeError("No extension in URL: " + url)

  if ext in savers:
    saver = savers[ext]
    saver(thing, url, **kwargs)
  else:
    message = "Unknown extension '{}', supports {}."
    raise RuntimeError(message.format(ext, loaders))
