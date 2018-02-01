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


"""Methods for displaying images from Numpy arrays."""

from __future__ import absolute_import, division, print_function

import base64
import io
import numpy as np
import PIL.Image


# For tests
_last_html_output = None

try:
  from IPython.core.display import display, HTML
  
  def _dispaly_html(html_str):
    global _last_html_output
    _last_html_output = html_str
    display(HTML(html_str))
  
except ImportError:
  print('IPython is no present, HTML output from '
        'lucid.misc.show will be ignored.')
  
  def _dispaly_html(html_str):
    global _last_html_output
    _last_html_output = html_str
    pass
  


def _prep_image(a, domain=(0, 1), w=None, verbose=True):
  """Normalize an image pixel values and width and convert to PIL.Image.

  Args:
    a: NumPy array representing the image
    domain: Domain of pixel values, inferred from min & max values if None
    w: width of output image, scaled using nearest neighbor interpolation.
      size unchanged if None

  Returns:
    normalized PIL.Image
  """
  # squeeze helps both with batch=1 and B/W
  a = np.squeeze(np.asarray(a))
  a = a.astype("float32")

  # if domain is None, we will guess if it should be 0.,1. or 0,255
  if domain is None:
    domain = (a.min(), a.max())
    if verbose:
        print("Inferring pixel value domain of ", domain)

  if a.min() < domain[0] or a.max() > domain[1]:
    if verbose:
        print("clipping domain from", (a.min(), a.max()), "to", domain)
    a = a.clip(*domain)
  a = (a - domain[0]) / (domain[1] - domain[0]) * 255
  a = np.uint8(a)

  im = PIL.Image.fromarray(a)
  if w is not None:
    aspect = float(im.size[0]) / im.size[1]
    h = int(w / aspect)
    im = im.resize((w, h), PIL.Image.NEAREST)
  return im


def _image_url(im, fmt="png", mode="data"):
  """Create a data URL representing an image from a PIL.Image.

  Args:
    im: PIL.Image
    fmt: Format of output image
    mode: presently only supports "data" for data URL

  Returns:
    URL representing image
  """
  # think about supporting saving to CNS
  # potential params: cns="tmp", name="foo.png"
  # think about saving to Cloud Storage

  if mode == "data":
    if fmt == "jpg":
      fmt = "jpeg"
    f = io.BytesIO()
    im.save(f, fmt, quality=95)
    return ("data:image/" + fmt + ";base64," 
            + base64.b64encode(f.getvalue()).decode('ascii'))
  else:
    assert False, "unsupported mode"
  # elif mode == "cns":
  #   cns.save()


def image(array, fmt="png", domain=(0, 1), w=None, verbose=True):
  """Display an image.

  Args:
    array: NumPy array representing the image
    fmt: Image format e.g. png, jpeg
    domain: Domain of pixel values, inferred from min & max values if None
    w: width of output image, scaled using nearest neighbor interpolation.
      size unchanged if None
"""
  im = _prep_image(array, domain, w, verbose=verbose)
  url = _image_url(im, fmt)
  _dispaly_html('<img src="' + url + '">')


def images(arrays, labels=None, fmt="png", domain=(0, 1), w=None,
           verbose=True):
  """Display a list of images with optional labels.

  Args:
    arrays: A list of NumPy arrays representing images
    labels: A list of strings to label each image.
      Defaults to show index if None
    fmt: Image format e.g. png, jpeg
    w: width of output image, scaled using nearest neighbor interpolation.
      size unchanged if None
    domain: Domain of pixel values, inferred from min & max values if None
  """

  s = '<div style="display:flex;flex-direction:row;">'
  for i, a in enumerate(arrays):
    im = _prep_image(a, domain, w, verbose=verbose)
    url = _image_url(im, fmt)
    label = labels[i] if labels is not None else i
    label = str(label)

    s += """<div style="margin-right:10px;">
              {label}<br/>
              <img src="{url}" style="margin-top:4px;">
            </div>""".format(label=label, url=url)
  s += "</div>"
  _dispaly_html(s)