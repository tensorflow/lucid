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

import base64
from cStringIO import StringIO
import numpy as np
import PIL.Image
import scipy.ndimage as nd
from IPython.core.display import display, HTML


def _dispaly_html(html_str):
    display(HTML(html_str))


def _prep_image(a, domain=(0, 1), w=None, verbose=True):
  """Normalize an image pixel values and width.

  Args:
    a: NumPy array representing the image
    domain: Domain of pixel values, inferred from min & max values if None
    w: width of output image, scaled using nearest neighbor interpolation.
      size unchanged if None

  Returns:
    normalized image array
  """
  # squeeze helps both with batch=1 and B/W
  a = np.squeeze(np.asarray(a))
  a = a.astype("float32")

  # if domain is None, we will guess if it should be 0.,1. or 0,255
  if domain is None:
    domain = (a.min(), a.max())
    if verbose:
        print "Inferring pixel value domain of ", domain

  if a.min() < domain[0] or a.max() > domain[1]:
    if verbose:
        print "clipping domain from", (a.min(), a.max()), "to", domain
    a = a.clip(*domain)
  a = (a - domain[0]) / (domain[1] - domain[0]) * 255
  a = np.uint8(a)

  if w is not None:
    original_w = min(a.shape[0], a.shape[1])
    ratio = w / float(original_w)
    if len(a.shape) == 2:
      a = nd.zoom(a, [ratio, ratio], order=0)
    else:
      a = nd.zoom(a, [ratio, ratio, 1], order=0)
  return a


def _image_url(a, fmt="png", mode="data"):
  """Create a data URL representing an image from a NumPy array.

  Args:
    a: NumPy array
    fmt: Format of output image
    mode: presently only supports "data" for data URL

  Returns:
    URL representing image
  """
  # think about supporting saving to CNS
  # potential params: cns="tmp", name="foo.png"
  # think about saving to Cloud Storage

  # infer image mode from shape
  if len(a.shape) == 2:
    image_mode = "L"
  else:
    depth = a.shape[-1]
    if depth == 3:
      image_mode = "RGB"
    elif depth == 4:
      image_mode = "RGBA"
    else:
      raise RuntimeError("""show.image only supports 2D, 3D & 4D images,
                         invalid shape: """ + str(a.shape))

  if mode == "data":
    f = StringIO()
    if fmt == "jpg":
      fmt = "jpeg"
    PIL.Image.fromarray(a, mode=image_mode).save(f, fmt)
    return "data:image/" + fmt + ";base64," + base64.b64encode(f.getvalue())
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
  array = _prep_image(array, domain, w, verbose=verbose)
  url = _image_url(array, fmt)
  _dispaly_html('<img src="' + url + '">')


def images(arrays, labels=None, fmt="png", domain=(0, 1), w=None, verbose=True):
  """Display a list of images with optional labels.

  Args:
    arrays: A list of NumPy arrays representing images
    labels: A list of strings to label each image.
      Defaults to show index if None
    fmt: Image format e.g. png, jpeg
    domain: Domain of pixel values, inferred from min & max values if None
    w: width of output image, scaled using nearest neighbor interpolation.
      size unchanged if None
  """

  s = '<div style="display:flex;flex-direction:row;">'
  for i, a in enumerate(arrays):
    a = _prep_image(a, domain, w, verbose=verbose)
    url = _image_url(a, fmt)
    label = labels[i] if labels is not None else i
    label = str(label)

    s += """<div style="margin-right:10px;">
              {label}<br/>
              <img src="{url}" style="margin-top:4px;">
            </div>""".format(label=label, url=url)
  s += "</div>"
  _dispaly_html(s)
