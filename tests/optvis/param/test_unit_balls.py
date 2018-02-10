from __future__ import absolute_import, division, print_function

import pytest

import tensorflow as tf
import logging
from lucid.optvis.param.unit_balls import unit_ball_L2, unit_ball_L_inf


learning_rate = .1
num_steps = 16
log = logging.getLogger(__name__)


@pytest.mark.xfail(reason="Unpredictably imprecise on Travis right now.")
@pytest.mark.parametrize("shape", [(3), (2, 5, 5)])
def test_unit_ball_L2(shape, eps=1e-6):
  """Tests that a L2 unit ball variable's norm stays roughly within 1.0.
  Note: only holds down to eps ~= 5e-7.
  """
  with tf.Session() as sess:
    unit_ball = unit_ball_L2(shape)
    unit_ball_L2_norm = tf.norm(unit_ball)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    objective = optimizer.minimize(-unit_ball)
    tf.global_variables_initializer().run()
    norm_value = unit_ball_L2_norm.eval()
    for i in range(num_steps):
      _, new_norm_value, unit_ball_value = sess.run([objective, unit_ball_L2_norm, unit_ball])
      log.info("L2: %s, Value: %s", new_norm_value, unit_ball_value)
      assert new_norm_value >= norm_value - eps
      assert new_norm_value <= 1.0 + eps
      norm_value = new_norm_value


@pytest.mark.xfail(reason="Unpredictably imprecise on Travis right now.")
@pytest.mark.parametrize("shape", [(3), (2, 5, 5)])
@pytest.mark.parametrize("precondition", [True, False])
def test_unit_ball_L_inf(shape, precondition, eps=1e-6):
  """Tests that a L infinity unit ball variables' stay roughly within 1.0.
  Note: only holds down to eps ~= 5e-7.
  """
  with tf.Session() as sess:
    unit_ball = unit_ball_L_inf(shape, precondition=precondition)
    unit_ball_max = tf.reduce_max(unit_ball)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    objective = optimizer.minimize(-unit_ball)
    tf.global_variables_initializer().run()
    for i in range(num_steps):
      _, unit_ball_max_value, unit_ball_value = sess.run([objective, unit_ball_max, unit_ball])
      log.info("Linf: %s, Value: %s", unit_ball_max_value, unit_ball_value)
      assert unit_ball_max_value <= 1.0 + eps
