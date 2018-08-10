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
import uuid

from lucid.optvis import param
from lucid.optvis.transform.utils import compose, angle2rads, rand_select
from lucid.optvis.transform.operators import _parameterized_flattened_homography


def pad(w, mode="REFLECT", constant_value=0.5):
    def inner(image_t):
        if constant_value == "uniform":
            constant_value_ = tf.random_uniform([], 0, 1)
        else:
            constant_value_ = constant_value
        return tf.pad(
            image_t,
            [(0, 0), (w, w), (w, w), (0, 0)],
            mode=mode,
            constant_values=constant_value_,
        )

    return inner


def jitter(d, seed=None):
    def inner(image_t):
        image_t = tf.convert_to_tensor(image_t, preferred_dtype=tf.float32)
        t_shp = tf.shape(image_t)
        crop_shape = tf.concat([t_shp[:-3], t_shp[-3:-1] - d, t_shp[-1:]], 0)
        crop = tf.random_crop(image_t, crop_shape, seed=seed)
        shp = image_t.get_shape().as_list()
        mid_shp_changed = [
            shp[-3] - d if shp[-3] is not None else None,
            shp[-2] - d if shp[-3] is not None else None,
        ]
        crop.set_shape(shp[:-3] + mid_shp_changed + shp[-1:])
        return crop

    return inner


# 2D only version
def scale(scales, seed=None):
    def inner(image_t):
        image_t = tf.convert_to_tensor(image_t, preferred_dtype=tf.float32)
        scale = rand_select(scales, seed=seed)
        shp = tf.shape(image_t)
        scale_shape = tf.cast(scale * tf.cast(shp[-3:-1], "float32"), "int32")
        return tf.image.resize_bilinear(image_t, scale_shape)

    return inner


def rotate(angles, units="degrees", seed=None):
    def inner(image_t):
        image_t = tf.convert_to_tensor(image_t, preferred_dtype=tf.float32)
        angle = rand_select(angles, seed=seed)
        angle_rad = angle2rads(angle, units)
        return tf.contrib.image.rotate(image_t, angle_rad)

    return inner


def homography(seed=None, interpolation="BILINEAR"):
    """Most general 2D transform that can replace all our spatial transforms.
    Consists of an affine transformation + a perspective projection.
    TODO: how should we pass all the parameters? Dict with defaults? Bunch of bools? Long list of arguments?
    """
    def inner(image_t):
        translation1_x = tf.truncated_normal([], stddev=4)
        translation1_y = tf.truncated_normal([], stddev=4)
        rotationAngleInRadians = angle2rads(tf.truncated_normal([], stddev=5.0))
        shearingAngleInRadians = angle2rads(tf.truncated_normal([], stddev=2.5))
        shear_x = tf.truncated_normal([], stddev=1e-2)
        shear_y = tf.truncated_normal([], stddev=1e-2)
        vanishing_point_x = tf.truncated_normal([], stddev=1e-4)
        vanishing_point_y = tf.truncated_normal([], stddev=1e-4)
        translation2_x = tf.truncated_normal([], stddev=2)
        translation2_y = tf.truncated_normal([], stddev=2)
        shape_xy = tf.shape(image_t)[1:3]

        transform_t = tf.py_func(
            _parameterized_flattened_homography,
            [
                translation1_x,
                translation1_y,
                rotationAngleInRadians,
                shearingAngleInRadians,
                shear_x,
                shear_y,
                vanishing_point_x,
                vanishing_point_y,
                translation2_x,
                translation2_y,
                shape_xy,
            ],
            (tf.float32,),
            stateful=False,
        )
        transformed_t = tf.contrib.image.transform(
            image_t, transform_t, interpolation=interpolation
        )
        return transformed_t

    return inner
