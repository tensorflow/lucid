import pytest

import tensorflow as tf
from lucid.misc.gradient_override import use_gradient

def test_use_gradient():
  def foo_grad(op, grad):
    return tf.constant(42), tf.constant(43)

  @use_gradient(foo_grad)
  def foo(x, y):
    return x + y

  with tf.Session().as_default() as sess:
    x = tf.constant(1.)
    y = tf.constant(2.)
    z = foo(x, y)
    grad_wrt_x = tf.gradients(z, x, [1.])[0]
    grad_wrt_y = tf.gradients(z, y, [1.])[0]
    assert grad_wrt_x.eval() == 42
    assert grad_wrt_y.eval() == 43


from lucid.misc.gradient_override import gradient_override_map

def test_gradient_override_map():

  def gradient_override(op, grad):
    return tf.constant(42)

  with tf.Session().as_default() as sess:
    global_step = tf.train.get_or_create_global_step()
    init_global_step = tf.variables_initializer([global_step])
    init_global_step.run()

    a = tf.constant(1.)
    standard_relu = tf.nn.relu(a)
    grad_wrt_a = tf.gradients(standard_relu, a, [1.])[0]
    with gradient_override_map({"Relu": gradient_override}):
      overriden_relu = tf.nn.relu(a)
      overriden_grad_wrt_a = tf.gradients(overriden_relu, a, [1.])[0]
    assert grad_wrt_a.eval() != overriden_grad_wrt_a.eval()
    assert overriden_grad_wrt_a.eval() == 42


from lucid.misc.redirected_relu_grad import redirected_relu_grad, redirected_relu6_grad

relu_examples = [
    (1., -1., 0.), (-1., -1., -1.),
    (1.,  1., 1.), (-1.,  1., -1.),
]
relu6_examples = relu_examples + [
    (1.,  7., 1.), (-1.,  7.,  0.),
]
nonls = [("Relu", tf.nn.relu, redirected_relu_grad, relu_examples),
         ("Relu6", tf.nn.relu6, redirected_relu6_grad, relu6_examples)]

@pytest.mark.parametrize("nonl_name,nonl,nonl_grad_override, examples", nonls)
def test_gradient_override_relu6_directionality(nonl_name, nonl,
    nonl_grad_override, examples):
  for incoming_grad, input, grad in examples:
    with tf.Session().as_default() as sess:
      global_step = tf.train.get_or_create_global_step()
      init_global_step = tf.variables_initializer([global_step])
      init_global_step.run()

      batched_shape = [1,1]
      incoming_grad_t = tf.constant(incoming_grad, shape=batched_shape)
      input_t = tf.constant(input, shape=batched_shape)
      with gradient_override_map({nonl_name: nonl_grad_override}):
        nonl_t = nonl(input_t)
        grad_wrt_input = tf.gradients(nonl_t, input_t, [incoming_grad_t])[0]
      assert (grad_wrt_input.eval() == grad).all()

@pytest.mark.parametrize("nonl_name,nonl,nonl_grad_override, examples", nonls)
def test_gradient_override_shutoff(nonl_name, nonl,
    nonl_grad_override, examples):
  for incoming_grad, input, grad in examples:
    with tf.Session().as_default() as sess:
      global_step_t = tf.train.get_or_create_global_step()
      global_step_init_op = tf.variables_initializer([global_step_t])
      global_step_init_op.run()
      global_step_assign_t = tf.assign(global_step_t, 17)
      sess.run(global_step_assign_t)

      # similar setup to test_gradient_override_relu6_directionality,
      # but we test that the gradient is *not* what we're expecting as after 16
      # steps the override is shut off
      batched_shape = [1,1]
      incoming_grad_t = tf.constant(incoming_grad, shape=batched_shape)
      input_t = tf.constant(input, shape=batched_shape)
      with gradient_override_map({nonl_name: nonl_grad_override}):
        nonl_t = nonl(input_t)
        grad_wrt_input = tf.gradients(nonl_t, input_t, [incoming_grad_t])[0]
      nonl_t_no_override = nonl(input_t)
      grad_wrt_input_no_override = tf.gradients(nonl_t_no_override, input_t, [incoming_grad_t])[0]
      assert (grad_wrt_input.eval() == grad_wrt_input_no_override.eval()).all()
