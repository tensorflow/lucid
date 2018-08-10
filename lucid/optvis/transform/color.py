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

"""Color tranformations you might want neural net visualizations to be robust to.

This module provides a variety of functions which stochastically transform a
tensorflow tensor. The functions are of the form:

  (config) => (tensor) => (stochastic transformed tensor)

"""

import tensorflow as tf
import math
from lucid.optvis.transform.utils import angle2rads


_RGB_TO_YIQ_MATRIX = [
    [0.299, 0.587, 0.114],
    [0.596, -0.274, -0.321],
    [0.211, -0.523, 0.311]
]

_YIQ_TO_RGB_MATRIX = [
    [1, 0.956, 0.621],
    [1, -0.272, -0.647],
    [1, -1.107, 1.705]
]


def contrast(lower=0.8, upper=1.2, seed=None):
    """Adjusts contrast independently per channel, like tf.image.adjust_contrast
    See https://www.tensorflow.org/api_docs/python/tf/image/adjust_contrast
    """

    if upper <= lower:
        raise ValueError('upper must be > lower.')

    if lower < 0:
        raise ValueError('lower must be non-negative.')

    def inner(image_t):
        contrast_factor = tf.random_uniform([], lower, upper, seed=seed)
        per_channel_means = tf.reduce_mean(image_t, axis=[-3, -2])
        return (image_t - per_channel_means) * contrast_factor + per_channel_means
    return inner


def hue(max_hue_rotation, unit="degrees", seed=None):
    """Adjusts hue by converting to YIQ and rotating around color axis."""

    angle_rad = angle2rads(max_hue_rotation, unit)

    if angle_rad > math.pi:
      raise ValueError('angle_range must be smaller than pi'
                       '--otherwise it just wraps around.')

    if angle_rad <= 0:
      raise ValueError('angle_range must be positive.')

    def rotation_matrix(angle_t):
      cos = tf.cos(angle_t)
      sin = tf.sin(angle_t)
      template = tf.stack([
       tf.ones([]), tf.zeros([]), tf.zeros([]),
       tf.zeros([]), cos, -sin,
       tf.zeros([]), sin, cos
      ])
      return tf.reshape(template, (3,3))

    def inner(image_t):
      rgb_to_yiq = tf.constant(_RGB_TO_YIQ_MATRIX)
      yiq_to_rgb = tf.constant(_YIQ_TO_RGB_MATRIX)
      shape = tf.shape(image_t)

      # truncated_normal cuts off at 2 stdev, so we set stdev to half our max
      hue_rotation_angle = tf.truncated_normal([], stddev=angle_rad/2.0, seed=seed)
      rotation = rotation_matrix(hue_rotation_angle)

      image_flat = tf.reshape(image_t, (-1, 3))
      image_t_yiq = tf.matmul(image_flat, rgb_to_yiq)
      image_t_rotated = tf.matmul(image_t_yiq, rotation)
      image_t_transformed = tf.matmul(image_t_rotated, yiq_to_rgb)
      return tf.reshape(image_t_transformed, shape)
    return inner


def saturation(scale_range, seed=None):

    if scale_range < 0:
        raise ValueError('scale_range must be non-negative.')

    def scale_matrix(scale_t):
      template = tf.stack([
       tf.ones([]), tf.zeros([]), tf.zeros([]),
       tf.zeros([]), scale_t, tf.zeros([]),
       tf.zeros([]), tf.zeros([]), scale_t
      ])
      return tf.reshape(template, (3,3))

    def inner(image_t):
      shape = tf.shape(image_t)

      rgb_to_yiq = tf.constant(_RGB_TO_YIQ_MATRIX)
      yiq_to_rgb = tf.constant(_YIQ_TO_RGB_MATRIX)

      scale_t = tf.truncated_normal([], mean=1.0, stddev=scale_range/2.0, seed=seed)
      scaling = scale_matrix(scale_t)

      image_flat = tf.reshape(image_t, (-1, 3))
      image_t_yiq = tf.matmul(image_flat, rgb_to_yiq)
      image_t_scaled = tf.matmul(image_t_yiq, scaling)
      transformed_image_t = tf.matmul(image_t_scaled, yiq_to_rgb)
      return tf.reshape(transformed_image_t, shape)
    return inner


def jitter(d, seed=None):
    """Offset individual channels by at most the given amount.
    NOT currently exported because in practice it made everything too saturated.
    """

    if d < 0:
        raise ValueError('d must be > 0.')

    def inner(image_t):
        image_t = tf.convert_to_tensor(image_t, preferred_dtype=tf.float32)
        batch, height, width, depth = image_t.shape
        channels = tf.unstack(image_t, axis=-1)
        crop_size = [batch, height-d, width-d]
        crops = [tf.random_crop(channel, crop_size, seed=seed)
                 for channel in channels]
        crop = tf.stack(crops, axis=-1)
        return crop
    return inner
