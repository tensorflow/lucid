"""Optimize within unit balls in TensorFlow.

In adverserial examples, one often wants to optize within a constrained ball.
This module makes this easy through functions like unit_ball_L2(), which
creates a tensorflow variable constrained within a L2 unit ball.

EXPERIMENTAL: Do not use for adverserial examples if you need to be confident
they are strong attacks. We are not yet confident in this code.
"""

import tensorflow as tf

from lucid.misc.gradient_override import use_gradient


def dot(a, b):
  return tf.reduce_sum(a * b)


def _constrain_L2_grad(op, grad):
  """Gradient for constrained optimization on an L2 unit ball.

  This function projects the gradient onto the ball if you are on the boundary
  (or outside!), but leaves it untouched if you are inside the ball.

  Args:
    op: the tensorflow op we're computing the gradient for.
    grad: gradient we need to backprop

  Returns:
    (projected if necessary) gradient.
  """
  inp = op.inputs[0]
  inp_norm = tf.norm(inp)
  unit_inp = inp / inp_norm

  grad_projection = dot(unit_inp, grad)
  parallel_grad = unit_inp * grad_projection

  is_in_ball = tf.less_equal(inp_norm, 1)
  is_pointed_inward = tf.less(grad_projection, 0)
  allow_grad = tf.logical_or(is_in_ball, is_pointed_inward)
  clip_grad = tf.logical_not(allow_grad)

  clipped_grad = tf.cond(clip_grad, lambda: grad - parallel_grad, lambda: grad)

  return clipped_grad


@use_gradient(_constrain_L2_grad)
def constrain_L2(x):
  return x / tf.maximum(1.0, tf.norm(x))


def unit_ball_L2(shape):
  """A tensorflow variable tranfomed to be constrained in a L2 unit ball.

  EXPERIMENTAL: Do not use for adverserial examples if you need to be confident
  they are strong attacks. We are not yet confident in this code.
  """
  x = tf.Variable(tf.zeros(shape))
  return constrain_L2(x)


def _constrain_L_inf_grad(precondition=True):

  def grad_f(op, grad):
    """Gradient for constrained preconditioned optimization on an L_inf unit
    ball.

    This function projects the gradient onto the ball if you are on the
    boundary (or outside!). It always preconditions the gradient so it is the
    direction of steepest descent under L_inf.

    Args:
      op: the tensorflow op we're computing the gradient for.
      grad: gradient we need to backprop

    Returns:
      (projected if necessary) preconditioned gradient.
    """
    inp = op.inputs[0]
    dim_at_edge = tf.greater_equal(tf.abs(inp), 1.0)
    dim_outward = tf.greater(inp * grad, 0.0)
    if precondition:
      grad = tf.sign(grad)

    return tf.where(
        tf.logical_and(dim_at_edge, dim_outward),
        tf.zeros(grad.shape),
        grad
      )
  return grad_f


@use_gradient(_constrain_L_inf_grad(precondition=True))
def constrain_L_inf_precondition(x):
  return x / tf.maximum(1.0, tf.abs(x))


@use_gradient(_constrain_L_inf_grad(precondition=False))
def constrain_L_inf(x):
  return x / tf.maximum(1.0, tf.abs(x))


def unit_ball_L_inf(shape, precondition=True):
  """A tensorflow variable tranfomed to be constrained in a L_inf unit ball.

  Note that this code also preconditions the gradient to go in the L_inf
  direction of steepest descent.

  EXPERIMENTAL: Do not use for adverserial examples if you need to be confident
  they are strong attacks. We are not yet confident in this code.
  """
  x = tf.Variable(tf.zeros(shape))
  if precondition:
    return constrain_L_inf_precondition(x)
  else:
    return constrain_L_inf(x)
