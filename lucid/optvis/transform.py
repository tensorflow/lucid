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


"""Tranformations you might want neural net visualizations to be robust to.

This module provides a variety of functions which stochastically transform a
tensorflow tensor. The functions are of the form:

  (config) => (tensor) => (stochastic transformed tensor)

"""

import tensorflow as tf
import numpy as np

from lucid.optvis import param


def jitter(d, seed=None):
  def inner(t_image):
    t_image = tf.convert_to_tensor(t_image, preferred_dtype=tf.float32)
    t_shp = tf.shape(t_image)
    crop_shape = tf.concat([t_shp[:-3], t_shp[-3:-1] - d, t_shp[-1:]], 0)
    crop = tf.random_crop(t_image, crop_shape, seed=seed)
    shp = t_image.get_shape().as_list()
    mid_shp_changed = [shp[-3] - d if shp[-3] is not None else None,
                       shp[-2] - d if shp[-3] is not None else None]
    crop.set_shape(shp[:-3] + mid_shp_changed + shp[-1:])
    return crop
  return inner


def pad(w, mode="REFLECT", constant_value=0.5):
  def inner(t_image):
    if constant_value == "uniform":
      constant_value_ = tf.random_uniform([], 0, 1)
    else:
      constant_value_ = constant_value
    return tf.pad(t_image, [(0,0), (w,w), (w,w), (0,0)], mode=mode, constant_values=constant_value_)
  return inner


# def random_scale(scales, seed=None):
#   def inner(t):
#     t = tf.convert_to_tensor(t, preferred_dtype="float32")
#     scale = _rand_select(scales, seed=seed)
#     shp = tf.shape(t)
#     scale_shape = tf.concat(
#         [shp[:-3], tf.cast(scale * tf.cast(shp[-3:-1], "float32"), "int32"), shp[-1:]], 0)
#     return resize_bilinear_nd(t, scale_shape)
#   return inner

# 2D only version
def random_scale(scales, seed=None):
  def inner(t):
    t = tf.convert_to_tensor(t, preferred_dtype=tf.float32)
    scale = _rand_select(scales, seed=seed)
    shp = tf.shape(t)
    scale_shape = tf.cast(scale * tf.cast(shp[-3:-1], "float32"), "int32")
    return tf.image.resize_bilinear(t, scale_shape)
  return inner


def random_rotate(angles, units="degrees", seed=None):
  def inner(t):
    t = tf.convert_to_tensor(t, preferred_dtype=tf.float32)
    angle = _rand_select(angles, seed=seed)
    angle = _angle2rads(angle, units)
    return tf.contrib.image.rotate(t, angle)
  return inner


def normalize_gradient(grad_scales=None):

  if grad_scales is not None:
    grad_scales = np.float32(grad_scales)

  op_name = "NormalizeGrad_" + hex(np.random.randint(1e6))[2:]
  @tf.RegisterGradient(op_name)
  def _NormalizeGrad(op, grad):
    grad_norm = tf.sqrt(tf.reduce_sum(grad**2, [1, 2, 3], keep_dims=True))
    if grad_scales is not None:
      grad *= grad_scales[:, None, None, None]
    return grad / grad_norm

  def inner(x):
    with x.graph.gradient_override_map({'Identity': op_name}):
      x = tf.identity(x)
    return x

  return inner


def compose(transforms):
  def inner(x):
    for transform in transforms:
      x = transform(x)
    return x
  return inner


def collapse_alpha_random(sd=0.5):
  def inner(t_image):
    rgb, a = t_image[..., :3], t_image[..., 3:4]
    rgb_shape = rgb.get_shape().as_list()
    rand_img = param.image_sample(rgb_shape, sd=sd)
    return a*rgb + (1-a)*rand_img
  return inner


def _rand_select(xs, seed=None):
  rand_n = tf.random_uniform((), 0, len(xs), "int32", seed=seed)
  return tf.constant(xs)[rand_n]


def _angle2rads(angle, units):
  angle = tf.cast(angle, "float32")
  if units.lower() == "degrees":
    angle = 3.14*angle/180.
  elif units.lower() in ["radians", "rads", "rad"]:
    angle = angle
  return angle
  
  
standard_transforms = [
    pad(12, mode='constant', constant_value=.5),
    jitter(8),
    random_scale([1 + (i-5)/50. for i in range(11)]),
    random_rotate(list(range(-10, 11)) + 5*[0]),
    jitter(4),
  ]
