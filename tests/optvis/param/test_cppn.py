from __future__ import absolute_import, division, print_function

import pytest

import numpy as np
import tensorflow as tf

import logging

from lucid.optvis.param.cppn import cppn


log = logging.getLogger(__name__)


@pytest.mark.slow
def test_cppn_fits_xor():

    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        cppn_param = cppn(16, num_output_channels=1)[0]

        def xor_objective(a):
            return -(
                tf.square(a[0, 0])
                + tf.square(a[-1, -1])
                + tf.square(1.0 - a[-1, 0])
                + tf.square(1.0 - a[0, -1])
            )

        loss_t = xor_objective(cppn_param)
        optimizer = tf.compat.v1.train.AdamOptimizer(0.01)
        objective = optimizer.minimize(loss_t)
        for try_i in range(3):
            tf.compat.v1.global_variables_initializer().run()
            # loss = loss_t.eval()
            for i in range(200):
                _, vis = sess.run([objective, cppn_param])
                close_enough = (
                    vis[0, 0] > .99
                    and vis[-1, -1] > .99
                    and vis[-1, 0] < .01
                    and vis[0, -1] < .01
                )
                if close_enough:
                    return
        assert False, "fitting XOR took more than 200 steps, failing test"
