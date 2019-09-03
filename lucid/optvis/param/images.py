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

"""High-level wrapper for paramaterizing images."""


import tensorflow as tf

from lucid.optvis.param.color import to_valid_rgb
from lucid.optvis.param.spatial import pixel_image, fft_image


def image(
    w,
    h=None,
    batch=None,
    sd=None,
    decorrelate=True,
    fft=True,
    alpha=False,
    channels=None,
):
    h = h or w
    batch = batch or 1
    ch = channels or (4 if alpha else 3)
    shape = [batch, h, w, ch]
    param_f = fft_image if fft else pixel_image
    t = param_f(shape, sd=sd)
    if channels:
        output = tf.nn.sigmoid(t)
    else:
        output = to_valid_rgb(t[..., :3], decorrelate=decorrelate, sigmoid=True)
        if alpha:
            a = tf.nn.sigmoid(t[..., 3:])
            output = tf.concat([output, a], -1)
    return output


def grayscale_image_rgb(*args, **kwargs):
    """Takes same arguments as image"""
    output = image(*args, channels=1, **kwargs)
    return tf.tile(output, (1, 1, 1, 3))
