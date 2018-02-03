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
import PIL.Image

from lucid.util.array_to_image import _infer_domain_from_array
from lucid.util.array_to_image import _normalize_array_and_convert_to_image


# create logger with module name, e.g. lucid.util.show
log = logging.getLogger(__name__)


_last_html_output = None
_last_data_output = None

try:
  from IPython.display import display as IPythonDisplay
  from IPython.display import HTML as IPythonHTML
  from IPython.display import Image as IPythonImage

  def _display_html(html_str):
    global _last_html_output
    _last_html_output = html_str
    IPythonDisplay(IPythonHTML(html_str))

  def _display_data(image_data, format):
    global _last_data_output
    _last_data_output = image_data
    IPythonDisplay(IPythonImage(data=image_data, format=format))

except ImportError:
  log.warn('IPython is not present, HTML output from lucid.util.show and '
           'lucid.util.show.image will be ignored.')

  def _display_html(html_str):
    global _last_html_output
    _last_html_output = html_str
    pass

  def _display_data(image_data, format):
    global _last_data_output
    _last_data_output = image_data
    pass


def _image_url(image, fmt='png', mode="data"):
  """Create a data URL representing an image from a PIL.Image.

  Args:
    image: a PIL.Image
    mode: presently only supports "data" for data URL

  Returns:
    URL representing image
  """
  # think about supporting saving to CNS
  # potential params: cns="tmp", name="foo.png"
  # think about saving to Cloud Storage

  if mode == "data":
    image_data = _data_from_image(image, fmt=fmt)
    base64_string = base64.b64encode(image_data).decode('ascii')
    return "data:image/" + fmt + ";base64," + base64_string
  else:
    raise ValueError("Unsupported mode '%s'", mode)


def _data_from_image(image, fmt='png', quality=95):
  """Serialize a PIL.Image"""
  image_bytes = BytesIO()
  image.save(image_bytes, fmt, quality=quality)
  return image_bytes.getvalue()


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
  domain = domain or _infer_domain_from_array(array)
  image = _normalize_array_and_convert_to_image(array, domain, w)
  image_data = _data_from_image(image, format)
  _display_data(image_data, format=format)


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

  s = '<div style="display:flex;flex-direction:row;">'
  for i, a in enumerate(arrays):
    domain = domain or _infer_domain_from_array(a)
    a = _normalize_array_and_convert_to_image(a, domain, w)
    url = _image_url(a)
    label = labels[i] if labels is not None else i
    label = str(label)

    s += """<div style="margin-right:10px;">
              {label}<br/>
              <img src="{url}" style="margin-top:4px;">
            </div>""".format(label=label, url=url)
  s += "</div>"
  _display_html(s)


def display(thing):
  """Display a nupmy array without having to specify what it represents.

  This module will attempt to infer how to display your tensor based on its
  rank, shape and dtype. rank 4 tensors will be displayed as image grids, rank
  2 and 3 tensors as images.
  """
  if isinstance(thing, np.ndarray):
    rank = len(thing.shape)
    if rank == 4:
      log.debug("Show is assuming rank 4 tensor to be a list of images.")
      images(thing)
    elif rank in (2, 3):
      log.debug("Show is assuming rank 2 or 3 tensor to be an image.")
      image(thing)
    else:
      log.warn("Show only supports numpy arrays of rank 2-4. Using repr().")
      print(repr(thing))
  elif isinstance(thing, (list, tuple)):
    log.debug("Show is assuming list or tuple to be a collection of images.")
    images(thing)
  else:
    log.warn("Show only supports numpy arrays so far. Using repr().")
    print(repr(thing))
