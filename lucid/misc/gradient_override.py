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

"""Easily declare tensorflow functions with custom gradients.

Normally, overriding gradients in TensorFlow requires you to register the
gradient function to a string and then use a with block with
graph.gradient_override_map().

There's a bunch of things about this that are annoying:

(1) You have to pick a string
(2) You can only use a string once, so if you're prototyping you need to
    generate a new name every time you modify the function.
(3) You have to make annoying with blocks all the time and it doesn't feel
    very functional.

This abstraction solves those issues. You no longer need to think about strings
or with blocks. Just use a single decorator and everything else will be
handled for you.

If you don't need to serialize your graph and the gradient override isn't
performance critical, you can use the high level `use_gradient()` decorator:

  @use_gradient(_foo_grad)
  def foo(x): ...

Otherwise, you can use use the lower level `gradient_override_map()`, a
convenience wrapper for `graph.gradient_override_map()`.
"""

from contextlib import contextmanager
import numpy as np
import tensorflow as tf


def register_to_random_name(grad_f):
  """Register a gradient function to a random string.

  In order to use a custom gradient in TensorFlow, it must be registered to a
  string. This is both a hassle, and -- because only one function can every be
  registered to a string -- annoying to iterate on in an interactive
  environemnt.

  This function registers a function to a unique random string of the form:

    {FUNCTION_NAME}_{RANDOM_SALT}

  And then returns the random string. This is a helper in creating more
  convenient gradient overrides.

  Args:
    grad_f: gradient function to register. Should map (op, grad) -> grad(s)

  Returns:
    String that gradient function was registered to.
  """
  grad_f_name = grad_f.__name__ + "_" + hex(np.random.randint(0, 1e10))[2:]
  tf.RegisterGradient(grad_f_name)(grad_f)
  return grad_f_name


@contextmanager
def gradient_override_map(override_dict):
  """Convenience wrapper for graph.gradient_override_map().

  This functions provides two conveniences over normal tensorflow gradient
  overrides: it auomatically uses the default graph instead of you needing to
  find the graph, and it automatically

  Example:

    def _foo_grad_alt(op, grad): ...

    with gradient_override({"Foo": _foo_grad_alt}):

  Args:
    override_dict: A dictionary describing how to override the gradient.
      keys: strings correponding to the op type that should have their gradient
        overriden.
      values: functions or strings registered to gradient functions

  """
  override_dict_by_name = {}
  for (op_name, grad_f) in override_dict.items():
    if isinstance(grad_f, str):
       override_dict_by_name[op_name] = grad_f
    else:
      override_dict_by_name[op_name] = register_to_random_name(grad_f)
  with tf.get_default_graph().gradient_override_map(override_dict_by_name):
    yield


def use_gradient(grad_f):
  """Decorator for easily setting custom gradients for TensorFlow functions.

  * DO NOT use this function if you need to serialize your graph.
  * This function will cause the decorated function to run slower.

  Example:

    def _foo_grad(op, grad): ...

    @use_gradient(_foo_grad)
    def foo(x1, x2, x3): ...

  Args:
    grad_f: function to use as gradient.

  Returns:
    A decorator to apply to the function you wish to override the gradient of.

  """
  grad_f_name = register_to_random_name(grad_f)

  def function_wrapper(f):
    def inner(*inputs):

      # TensorFlow only supports (as of writing) overriding the gradient of
      # individual ops. In order to override the gardient of `f`, we need to
      # somehow make it appear to be an individual TensorFlow op.
      #
      # Our solution is to create a PyFunc that mimics `f`.
      #
      # In particular, we construct a graph for `f` and run it, then use a
      # stateful PyFunc to stash it's results in Python. Then we have another
      # PyFunc mimic it by taking all the same inputs and returning the stashed
      # output.
      #
      # I wish we could do this without PyFunc, but I don't see a way to have
      # it be fully general.

      state = {"out_value": None}

      # First, we need to run `f` and store it's output.

      out = f(*inputs)

      def store_out(out_value):
        """Store the value of out to a python variable."""
        state["out_value"] = out_value

      store_name = "store_" + f.__name__
      store = tf.py_func(store_out, [out], (), stateful=True, name=store_name)

      # Next, we create the mock function, with an overriden gradient.
      # Note that we need to make sure store gets evaluated before the mock
      # runs.

      def mock_f(*inputs):
        """Mimic f by retrieving the stored value of out."""
        return state["out_value"]

      with tf.control_dependencies([store]):
        with gradient_override_map({"PyFunc": grad_f_name}):
          mock_name = "mock_" + f.__name__
          mock_out = tf.py_func(mock_f, inputs, out.dtype, stateful=True,
                                name=mock_name)
          mock_out.set_shape(out.get_shape())

      # Finally, we can return the mock.

      return mock_out
    return inner
  return function_wrapper
