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


"""This module provides methods for paramaterizing images,

We represent paramterizations as functions of the form:

  () => TensorFlow Tensor

This means they can be declared outside the rendering function, but then
construct a graph within the session the render function declares.

"""

import numpy as np
import tensorflow as tf

from lucid.optvis.resize_bilinear_nd import resize_bilinear_nd


# def fft_tensor(shape, scale_freqs=True):


color_correlation_cholesky = np.asarray([[0.28, 0.00, 0.00],
                                         [0.25, 0.12, 0.00],
                                         [0.23, 0.14, 0.14]]).astype("float32")
max_norm_cholesky = np.max(np.linalg.norm(color_correlation_cholesky, axis=0))

color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")
max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))


def image(w, h=None, batch=None, sd=None, decorrelate=False, fft=False, alpha=False):
  h = h or w
  batch = batch or 1
  channels = 4 if alpha else 3
  shape = [batch, w, h, channels]
  param_f = fft_image if fft else naive
  t = param_f(shape, sd=sd)
  if alpha:
    rgb = rgb_sigmoid(t[..., :3], decorrelate=decorrelate)
    a = tf.nn.sigmoid(t[..., 3:])
    return tf.concat([rgb, a], -1)
  else:
    return rgb_sigmoid(t, decorrelate=decorrelate)


def naive_image(w, h=None, batch=None, sd=None, decorrelate=False):
  h = h or w
  batch = batch or 1
  t = naive([batch, w, h, 3], sd=sd)
  return rgb_sigmoid(t, decorrelate=decorrelate)


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


pyramid = laplacian_pyramid


def lowres_tensor(shape, underlying_shape, offset=None, sd=None):
  """Produces a tensor paramaterized by a interpolated lower resolution tensor.

  This is like what is done in a laplacian pyramid, but a bit more general. It
  can be a powerful way to describe images.

  Args:
    shape: desired shape of resulting tensor
    underlying_shape: shape of the tensor being resized into final tensor
    offset: Describes how to offset the interpolated vector (like phase in a
      Fourier transform). If None, apply no offset. If a scalar, apply the same
      offset to each dimension; if a list use each entry for each dimension.
      If a int, offset by that much. If False, do not offset. If True, offset by
      half the ratio between shape and underlying shape (analagous to 90
      degrees).
    sd: Standard deviation of initial tensor variable.

  Returns:
    A tensor paramaterized by a lower resolution tensorflow variable.
  """
  sd = sd or 0.01
  init_val = sd*np.random.randn(*underlying_shape).astype("float32")
  underlying_t = tf.Variable(init_val)
  t = resize_bilinear_nd(underlying_t, shape)
  if offset is not None:
    # Deal with non-list offset
    if not isinstance(offset, list):
      offset = len(shape)*[offset]
    # Deal with the non-int offset entries
    for n in range(len(offset)):
      if offset[n] is True:
        offset[n] = shape[n]/underlying_shape[n]/2
      if offset[n] is False:
        offset[n] = 0
      offset[n] = int(offset[n])
    # Actually apply offset by padding and then croping off the excess.
    padding = [(pad, 0) for pad in offset]
    t = tf.pad(t, padding, "SYMMETRIC")
    begin = len(shape)*[0]
    t = tf.slice(t, begin, shape)
  return t


def _linear_decorelate_color(t):
  t_flat = tf.reshape(t, [-1, 3])
  color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
  t_flat = tf.matmul(t_flat, color_correlation_normalized.T)
  t = tf.reshape(t_flat, tf.shape(t))
  return t


def svd_color_clipped(t):
  mean = [0.48, 0.46, 0.41]
  t = mean + _linear_decorelate_color(t)
  t = tf.clip_by_value(t, 0, 1)
  return t


def rgb_sigmoid(t, decorrelate=False):
  t = t[..., :3]
  if decorrelate:
    t = _linear_decorelate_color(t)
  return tf.nn.sigmoid(t)


def fancy_colors(t):
  def channel_vec(v):
    return np.asarray(v).reshape([1, 1, 1, 3])

  logits  = t[..., 0:3]
  logits += t[..., 3:4] * channel_vec([1, 1, 0])
  logits += t[..., 4:5] * channel_vec([1, 0, 1])
  logits += t[..., 5:6] * channel_vec([0, 1, 1])
  H_image = tf.nn.sigmoid(logits)

  L_logit = t[..., 6:7] * channel_vec([1, 1, 1])
  L = tf.nn.sigmoid(L_logit)
  S_logit = t[..., 7:8]
  S = tf.nn.sigmoid(S_logit)

  return (1-S)*L + S*H_image


def multi_interpolation_basis(n_objectives=6, n_interp_steps=5, width=128,
                              channels=3):
  """A paramaterization for interpolating between each pair of N objectives.

  Sometimes you want to interpolate between optimizing a bunch of objectives,
  in a paramaterization that encourages images to align.

  Args:
    n_objectives: number of objectives you want interpolate between
    n_interp_steps: number of interpolation steps
    width: width of intepolated images
    channel

  Returns:
    A [n_objectives, n_objectives, n_interp_steps, width, width, channel]
    shaped tensor, t, where the final [width, width, channel] should be
    seen as images, such that the following properties hold:

     t[a, b]    = t[b, a, ::-1]
     t[a, i, 0] = t[a, j, 0] for all i, j
     t[a, a, i] = t[a, a, j] for all i, j
     t[a, b, i] = t[b, a, -i] for all i

  """
  N, M, W, Ch = n_objectives, n_interp_steps, width, channels

  const_term = sum([lowres_tensor([W, W, Ch], [W/k, W/k, Ch])
                    for k in [1, 2, 4, 8]])
  const_term = tf.reshape(const_term, [1, 1, 1, W, W, Ch])

  example_interps = [
      sum([lowres_tensor([M, W, W, Ch], [2, W/k, W/k, Ch])
           for k in [1, 2, 4, 8]])
      for _ in range(N)]

  example_basis = []
  for n in range(N):
    col = []
    for m in range(N):
      interp = example_interps[n] + example_interps[m][::-1]
      col.append(interp)
    example_basis.append(col)

  interp_basis = []
  for n in range(N):
    col = [interp_basis[m][N-n][::-1] for m in range(n)]
    col.append(tf.zeros([M, W, W, 3]))
    for m in range(n+1, N):
      interp = sum([lowres_tensor([M, W, W, Ch], [M, W/k, W/k, Ch])
                    for k in [1, 2]])
      col.append(interp)
    interp_basis.append(col)

  basis = []
  for n in range(N):
    col_ex = tf.stack(example_basis[n])
    col_in = tf.stack(interp_basis[n])
    basis.append(col_ex + col_in)
  basis = tf.stack(basis)

  return basis + const_term



def image_sample(shape, sd=0.2, decay_power=1, decorrelate=True):
  b, h, w, ch = shape
  assert ch == 3
  imgs = []
  for _ in range(b):
    freqs = _rfft2d_freqs(h, w)
    fh, fw = freqs.shape
    spectrum_float = tf.random_normal([2, ch, fh, fw], stddev=sd, dtype=tf.float32)
    spectrum = tf.complex(spectrum_float[0], spectrum_float[1])
    spectrum_scale = 1.0 / np.maximum(freqs, 1.0/max(h, w))**decay_power
    # Scale the spectrum by the square-root of the number of pixels
    # to get a unitary transformation. This allows to use similar
    # leanring rates to pixel-wise optimisation.
    spectrum_scale *= np.sqrt(w*h)
    scaled_spectrum = spectrum * spectrum_scale
    img = tf.spectral.irfft2d(scaled_spectrum)
    # in case of odd input dimension we cut off the additional pixel
    # we get from irfft2d length computation
    img = img[:ch, :h, :w]
    img = tf.transpose(img, [1, 2, 0])
    imgs.append(img)
  return rgb_sigmoid(tf.stack(imgs)/4., decorrelate=decorrelate)
