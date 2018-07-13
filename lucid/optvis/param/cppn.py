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

"""Compositional Pattern Producing Networks for use as image parameterizations."""


import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim


def _composite_activation(x, biased=True):
    x = tf.atan(x)
    if biased:
        # Biased Coefficients computed by:
        #   def rms(x):
        #     return np.sqrt((x*x).mean())
        #   a = np.arctan(np.random.normal(0.0, 1.0, 10**6))
        #   print(rms(a), rms(a*a))
        coeffs = (.67, .6)
        means = (.0, .0)
    else:
        # Unbiased Coefficients computed by:
        #   a = np.arctan(np.random.normal(0.0, 1.0, 10**6))
        #   aa = a*a
        #   print(a.std(), aa.mean(), aa.std())
        coeffs = (.67, .396)
        means = (.0, .45)
    composite = [(x - means[0]) / coeffs[0], (x * x - means[1]) / coeffs[1]]
    return tf.concat(composite, -1)


def _relu_normalized_activation(x):
    x = tf.nn.relu(x)
    # Coefficients computed by:
    #   a = np.random.normal(0.0, 1.0, 10**6)
    #   a = np.maximum(a, 0.0)
    #   print(a.mean(), a.std())
    return (x - 0.40) / 0.58


def cppn(
    width,
    batch=1,
    num_output_channels=3,
    num_hidden_channels=24,
    num_layers=8,
    activation_func=_composite_activation,
    normalize=False,
):
    """Compositional Pattern Producing Network

    Args:
      width: width of resulting image, equals height
      batch: batch dimension of output, note that all params share the same weights!
      num_output_channels:
      num_hidden_channels:
      num_layers:
      activation_func:
      normalize:

    Returns:
      The collapsed shape, represented as a list.
    """
    r = 3.0 ** 0.5  # std(coord_range) == 1.0
    coord_range = tf.linspace(-r, r, width)
    y, x = tf.meshgrid(coord_range, coord_range, indexing="ij")
    net = tf.stack([tf.stack([x, y], -1)] * batch, 0)

    with slim.arg_scope(
        [slim.conv2d],
        kernel_size=[1, 1],
        activation_fn=None,
        weights_initializer=tf.initializers.variance_scaling(),
        biases_initializer=tf.initializers.random_normal(0.0, 0.1),
    ):
        for i in range(num_layers):
            x = slim.conv2d(net, num_hidden_channels)
            if normalize:
                x = slim.instance_norm(x)
            net = activation_func(x)
        rgb = slim.conv2d(
            net,
            num_output_channels,
            activation_fn=tf.nn.sigmoid,
            weights_initializer=tf.zeros_initializer(),
        )
    return rgb
