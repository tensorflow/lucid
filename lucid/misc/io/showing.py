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

"""Methods for displaying images from Numpy arrays."""

from __future__ import absolute_import, division, print_function

from io import BytesIO
import base64
import logging
import numpy as np
import IPython.display

from lucid.misc.io.serialize_array import serialize_array


# create logger with module name, e.g. lucid.misc.io.showing
log = logging.getLogger(__name__)


def _display_html(html_str):
  IPython.display.display(IPython.display.HTML(html_str))


def _image_url(array, fmt='png', mode="data", quality=70, domain=None):
  """Create a data URL representing an image from a PIL.Image.

  Args:
    image: a numpy
    mode: presently only supports "data" for data URL

  Returns:
    URL representing image
  """
  # TODO: think about supporting saving to CNS, potential params: cns, name
  # TODO: think about saving to Cloud Storage
  supported_modes = ("data")
  if mode not in supported_modes:
    message = "Unsupported mode '%s', should be one of '%s'."
    raise ValueError(message, mode, supported_modes)

  image_data = serialize_array(array, fmt=fmt, quality=quality)
  base64_byte_string = base64.b64encode(image_data).decode('ascii')
  return "data:image/" + fmt.upper() + ";base64," + base64_byte_string


# public functions


def image(array, domain=None, w=None, format='png'):
  """Display an image.

  Args:
    array: NumPy array representing the image
    fmt: Image format e.g. png, jpeg
    domain: Domain of pixel values, inferred from min & max values if None
    w: width of output image, scaled using nearest neighbor interpolation.
      size unchanged if None
  """
  data_url = _image_url(array, domain=domain)
  html = '<img src=\"' + data_url + '\">'
  _display_html(html)


def images(arrays, labels=None, domain=None, w=None):
  """Display a list of images with optional labels.

  Args:
    arrays: A list of NumPy arrays representing images
    labels: A list of strings to label each image.
      Defaults to show index if None
    domain: Domain of pixel values, inferred from min & max values if None
    w: width of output image, scaled using nearest neighbor interpolation.
      size unchanged if None
  """

  s = '<div style="display: flex; flex-direction: row;">'
  for i, array in enumerate(arrays):
    url = _image_url(array)
    label = labels[i] if labels is not None else i
    s += """<div style="margin-right:10px;">
              {label}<br/>
              <img src="{url}" style="margin-top:4px;">
            </div>""".format(label=label, url=url)
  s += "</div>"
  _display_html(s)


def show(thing, domain=(0, 1)):
  """Display a nupmy array without having to specify what it represents.

  This module will attempt to infer how to display your tensor based on its
  rank, shape and dtype. rank 4 tensors will be displayed as image grids, rank
  2 and 3 tensors as images.
  """
  if isinstance(thing, np.ndarray):
    rank = len(thing.shape)
    if rank == 4:
      log.debug("Show is assuming rank 4 tensor to be a list of images.")
      images(thing, domain=domain)
    elif rank in (2, 3):
      log.debug("Show is assuming rank 2 or 3 tensor to be an image.")
      image(thing, domain=domain)
    else:
      log.warn("Show only supports numpy arrays of rank 2-4. Using repr().")
      print(repr(thing))
  elif isinstance(thing, (list, tuple)):
    log.debug("Show is assuming list or tuple to be a collection of images.")
    images(thing, domain=domain)
  else:
    log.warn("Show only supports numpy arrays so far. Using repr().")
    print(repr(thing))
