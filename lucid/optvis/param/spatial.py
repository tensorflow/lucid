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

import warnings

import numpy as np
import tensorflow as tf

from lucid.optvis.param.lowres import lowres_tensor


def pixel_image(shape, sd=None, init_val=None):
    """A naive, pixel-based image parameterization.
    Defaults to a random initialization, but can take a supplied init_val argument
    instead.

    Args:
      shape: shape of resulting image, [batch, width, height, channels].
      sd: standard deviation of param initialization noise.
      init_val: an initial value to use instead of a random initialization. Needs
        to have the same shape as the supplied shape argument.

    Returns:
      tensor with shape from first argument.
    """
    if sd is not None and init_val is not None:
        warnings.warn(
            "`pixel_image` received both an initial value and a sd argument. Ignoring sd in favor of the supplied initial value."
        )

    sd = sd or 0.01
    init_val = init_val or np.random.normal(size=shape, scale=sd).astype(np.float32)
    return tf.Variable(init_val)


def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""

    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)


def fft_image(shape, sd=None, decay_power=1):
    """An image paramaterization using 2D Fourier coefficients."""

    sd = sd or 0.01
    batch, h, w, ch = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (2, batch, ch) + freqs.shape

    init_val = np.random.normal(size=init_val_size, scale=sd).astype(np.float32)
    spectrum_real_imag_t = tf.Variable(init_val)
    spectrum_t = tf.complex(spectrum_real_imag_t[0], spectrum_real_imag_t[1])

    # Scale the spectrum. First normalize energy, then scale by the square-root
    # of the number of pixels to get a unitary transformation.
    # This allows to use similar leanring rates to pixel-wise optimisation.
    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale *= np.sqrt(w * h)
    scaled_spectrum_t = scale * spectrum_t

    # convert complex scaled spectrum to shape (h, w, ch) image tensor
    # needs to transpose because irfft2d returns channels first
    image_t = tf.transpose(tf.signal.irfft2d(scaled_spectrum_t), (0, 2, 3, 1))

    # in case of odd spatial input dimensions we need to crop
    image_t = image_t[:batch, :h, :w, :ch]
    image_t = image_t / 4.0  # TODO: is that a magic constant?
    return image_t


def laplacian_pyramid_image(shape, n_levels=4, sd=None):
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
        k = 2 ** n
        pyramid += lowres_tensor(shape, batch_dims + (w // k, h // k, ch), sd=sd)
    return pyramid


def bilinearly_sampled_image(texture, uv):
    """Build bilinear texture sampling graph.

    Coordinate transformation rules match OpenGL GL_REPEAT wrapping and GL_LINEAR
    interpolation modes.

    Args:
      texture: [tex_h, tex_w, channel_n] tensor.
      uv: [frame_h, frame_h, 2] tensor with per-pixel UV coordinates in range [0..1]

    Returns:
      [frame_h, frame_h, channel_n] tensor with per-pixel sampled values.
    """
    h, w = tf.unstack(tf.shape(texture)[:2])
    u, v = tf.split(uv, 2, axis=-1)
    v = 1.0 - v  # vertical flip to match GL convention
    u, v = u * tf.to_float(w) - 0.5, v * tf.to_float(h) - 0.5
    u0, u1 = tf.floor(u), tf.ceil(u)
    v0, v1 = tf.floor(v), tf.ceil(v)
    uf, vf = u - u0, v - v0
    u0, u1, v0, v1 = map(tf.to_int32, [u0, u1, v0, v1])

    def sample(u, v):
        vu = tf.concat([v % h, u % w], axis=-1)
        return tf.gather_nd(texture, vu)

    s00, s01 = sample(u0, v0), sample(u0, v1)
    s10, s11 = sample(u1, v0), sample(u1, v1)
    s0 = s00 * (1.0 - vf) + s01 * vf
    s1 = s10 * (1.0 - vf) + s11 * vf
    s = s0 * (1.0 - uf) + s1 * uf
    return s


# Deprecations


def naive(shape, sd=None):
    warnings.warn(
        "`naive` has been renamed `pixel_image` for clarity.", DeprecationWarning
    )
    return pixel_image(shape, sd)


def laplacian_pyramid(shape, n_levels=4, sd=None):
    warnings.warn(
        "`laplacian_pyramid` has been renamed `laplacian_pyramid_image` for clarity.",
        DeprecationWarning,
    )
    return laplacian_pyramid_image(shape, n_levels=n_levels, sd=sd)


def sample_bilinear(texture, uv):
    warnings.warn(
        "`sample_bilinear` has been renamed `bilinearly_sampled_image` for clarity.",
        DeprecationWarning,
    )
    return bilinearly_sampled_image(texture, uv)
