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

import numpy as np
import tensorflow as tf

from lucid.optvis.param.lowres import lowres_tensor


def naive(shape, sd=None):
  return lowres_tensor(shape, shape, sd=sd)


def _rfft2d_freqs(h, w):
  """Compute 2d spectrum frequences."""
  fy = np.fft.fftfreq(h)[:, None]
  # when we have an odd input dimension we need to keep one additional
  # frequency and later cut off 1 pixel
  if w % 2 == 1:
    fx = np.fft.fftfreq(w)[:w//2+2]
  else:
    fx = np.fft.fftfreq(w)[:w//2+1]
  return np.sqrt(fx*fx + fy*fy)


def fft_image(shape, sd=None, decay_power=1):
  b, h, w, ch = shape
  imgs = []
  for _ in range(b):
    freqs = _rfft2d_freqs(h, w)
    fh, fw = freqs.shape
    sd = sd or 0.01
    init_val = sd*np.random.randn(2, ch, fh, fw).astype("float32")
    spectrum_var = tf.Variable(init_val)
    spectrum = tf.complex(spectrum_var[0], spectrum_var[1])
    spertum_scale = 1.0 / np.maximum(freqs, 1.0/max(h, w))**decay_power
    # Scale the spectrum by the square-root of the number of pixels
    # to get a unitary transformation. This allows to use similar
    # leanring rates to pixel-wise optimisation.
    spertum_scale *= np.sqrt(w*h)
    scaled_spectrum = spectrum * spertum_scale
    img = tf.spectral.irfft2d(scaled_spectrum)
    # in case of odd input dimension we cut off the additional pixel
    # we get from irfft2d length computation
    img = img[:ch, :h, :w]
    img = tf.transpose(img, [1, 2, 0])
    imgs.append(img)
  return tf.stack(imgs)/4.


def laplacian_pyramid(shape, n_levels=4, sd=None):
  """Simple laplacian pyramid paramaterization of an image.

  For more flexibility, use a sum of lowres_tensor()s.

  Args:
    shape: shape of resulting image, [batch, width, height, channels].
    n_levels: number of levels of laplacian pyarmid.
    sd: standard deviation of param initialization.

  Returns:
    tensor with shape from first argument.
  """
  batch_dims = shape[:-3]
  w, h, ch = shape[-3:]
  pyramid = 0
  for n in range(n_levels):
    k = 2**n
    pyramid += lowres_tensor(shape, batch_dims + [w // k, h // k, ch], sd=sd)
  return pyramid
