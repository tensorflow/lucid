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

"""Functions for transforming and constraining color channels."""


import numpy as np
import tensorflow as tf

from lucid.optvis.param.unit_balls import constrain_L_inf

color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")
max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

color_mean = [0.48, 0.46, 0.41]       


def _linear_decorelate_color(t):
  """Multiply input by sqrt of emperical (ImageNet) color correlation matrix.
  
  If you interpret t's innermost dimension as describing colors in a
  decorrelated version of the color space (which is a very natural way to
  describe colors -- see discussion in Feature Visualization article) the way
  to map back to normal colors is multiply the square root of your color
  correlations. 
  """
  # check that inner dimension is 3?
  t_flat = tf.reshape(t, [-1, 3])
  color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
  t_flat = tf.matmul(t_flat, color_correlation_normalized.T)
  t = tf.reshape(t_flat, tf.shape(t))
  return t


def to_valid_rgb(t, decorrelate=False, sigmoid=True):
  """Transform inner dimension of t to valid rgb colors.
  
  In practice this consistes of two parts: 
  (1) If requested, transform the colors from a decorrelated color space to RGB.
  (2) Constrain the color channels to be in [0,1], either using a sigmoid
      function or clipping.
  
  Args:
    t: input tensor, innermost dimension will be interpreted as colors
      and transformed/constrained.
    decorrelate: should the input tensor's colors be interpreted as coming from
      a whitened space or not?
    sigmoid: should the colors be constrained using sigmoid (if True) or
      clipping (if False).
  
  Returns:
    t with the innermost dimension transformed.
  """
  if decorrelate:
    t = _linear_decorelate_color(t)
  if decorrelate and not sigmoid:
    t += color_mean
  if sigmoid:
    return tf.nn.sigmoid(t)
  else:
    return constrain_L_inf(2*t-1)/2 + 0.5
