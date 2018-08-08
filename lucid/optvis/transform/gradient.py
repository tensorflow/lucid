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
from __future__ import absolute_import, division, print_function

import tensorflow as tf


def normalize(grad_scales=None):

    if grad_scales is not None:
        grad_scales = np.float32(grad_scales)

    op_name = "NormalizeGrad_" + str(uuid.uuid4())

    @tf.RegisterGradient(op_name)
    def _NormalizeGrad(op, grad):
        grad_norm = tf.sqrt(tf.reduce_sum(grad ** 2, [1, 2, 3], keepdims=True))
        if grad_scales is not None:
            grad *= grad_scales[:, None, None, None]
        return grad / grad_norm

    def inner(x):
        with x.graph.gradient_override_map({"Identity": op_name}):
            x = tf.identity(x)
        return x

    return inner
