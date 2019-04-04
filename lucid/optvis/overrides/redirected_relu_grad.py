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

"""Redirected ReLu Gradient Overrides

When we visualize ReLU networks, the initial random input we give the model may
not cause the neuron we're visualizing to fire at all. For a ReLU neuron, this
means that no gradient flow backwards and the visualization never takes off.
One solution would be to find the pre-ReLU tensor, but that can be tedious.

These functions provide a more convenient solution: temporarily override the
gradient of ReLUs to allow gradient to flow back through the ReLU -- even if it
didn't activate and had a derivative of zero -- allowing the visualization
process to get started. These functions override the gradient for at most 16
steps. Thus, you need to initialize `global_step` before using these functions.

Usage:
```python
from lucid.misc.gradient_override import gradient_override_map
from lucid.misc.redirected_relu_grad import redirected_relu_grad

...
global_step_t = tf.train.get_or_create_global_step()
init_global_step_op = tf.variables_initializer([global_step_t])
init_global_step_op.run()
...

with gradient_override_map({'Relu': redirected_relu_grad}):
  model.import_graph(...)
```

Discussion:
ReLus block the flow of the gradient during backpropagation when their input is
negative. ReLu6s also do so when the input is larger than 6. These overrides
change this behavior to allow gradient pushing the input into a desired regime
between these points.

(This override first checks if the entire gradient would be blocked, and only
changes it in that case. It does this check independently for each batch entry.)

In effect, this replaces the relu gradient with the following:

Regime       | Effect
============================================================
 0 <= x <= 6 | pass through gradient
 x < 0       | pass through gradient pushing the input up
 x > 6       | pass through gradient pushing the input down

Or visually:

  ReLu:                     |   |____________
                            |  /|
                            | / |
                ____________|/  |
                            0   6

  Override:     ------------|   |------------
                  allow  ->       <-  allow

Our implementations contains one extra complication:
tf.train.Optimizer performs gradient _descent_, so in the update step the
optimizer changes values in the opposite direction of the gradient. Thus, the
sign of the gradient in our overrides has the opposite of the intuitive effect:
negative gradient pushes the input up, positive pushes it down.
Thus, the code below only allows _negative_ gradient when the input is already
negative, and allows _positive_ gradient when the input is already above 6.


[0] That is because many model architectures don't provide easy access
to pre-relu tensors. For example, GoogLeNet's mixed__ layers are passed through
an activation function before being concatenated. We are still interested in the
entire concatenated layer, we would just like to skip the activation function.
"""

import tensorflow as tf


def redirected_relu_grad(op, grad):
  assert op.type == "Relu"
  x = op.inputs[0]

  # Compute ReLu gradient
  relu_grad = tf.where(x < 0., tf.zeros_like(grad), grad)

  # Compute redirected gradient: where do we need to zero out incoming gradient
  # to prevent input going lower if its already negative
  neg_pushing_lower = tf.logical_and(x < 0., grad > 0.)
  redirected_grad = tf.where(neg_pushing_lower, tf.zeros_like(grad), grad)

  # Ensure we have at least a rank 2 tensor, as we expect a batch dimension
  assert_op = tf.Assert(tf.greater(tf.rank(relu_grad), 1), [tf.rank(relu_grad)])
  with tf.control_dependencies([assert_op]):
    # only use redirected gradient where nothing got through original gradient
    batch = tf.shape(relu_grad)[0]
    reshaped_relu_grad = tf.reshape(relu_grad, [batch, -1])
    relu_grad_mag = tf.norm(reshaped_relu_grad, axis=1)
  result_grad = tf.where(relu_grad_mag > 0., relu_grad, redirected_grad)

  global_step_t =tf.train.get_or_create_global_step()
  return_relu_grad = tf.greater(global_step_t, tf.constant(16, tf.int64))

  return tf.where(return_relu_grad, relu_grad, result_grad)


def redirected_relu6_grad(op, grad):
  assert op.type == "Relu6"
  x = op.inputs[0]

  # Compute ReLu gradient
  relu6_cond = tf.logical_or(x < 0., x > 6.)
  relu_grad = tf.where(relu6_cond, tf.zeros_like(grad), grad)

  # Compute redirected gradient: where do we need to zero out incoming gradient
  # to prevent input going lower if its already negative, or going higher if
  # already bigger than 6?
  neg_pushing_lower = tf.logical_and(x < 0., grad > 0.)
  pos_pushing_higher = tf.logical_and(x > 6., grad < 0.)
  dir_filter = tf.logical_or(neg_pushing_lower, pos_pushing_higher)
  redirected_grad = tf.where(dir_filter, tf.zeros_like(grad), grad)

  # Ensure we have at least a rank 2 tensor, as we expect a batch dimension
  assert_op = tf.Assert(tf.greater(tf.rank(relu_grad), 1), [tf.rank(relu_grad)])
  with tf.control_dependencies([assert_op]):
    # only use redirected gradient where nothing got through original gradient
    batch = tf.shape(relu_grad)[0]
    reshaped_relu_grad = tf.reshape(relu_grad, [batch, -1])
    relu_grad_mag = tf.norm(reshaped_relu_grad, axis=1)
  result_grad =  tf.where(relu_grad_mag > 0., relu_grad, redirected_grad)

  global_step_t = tf.train.get_or_create_global_step()
  return_relu_grad = tf.greater(global_step_t, tf.constant(16, tf.int64))

  return tf.where(return_relu_grad, relu_grad, result_grad)
