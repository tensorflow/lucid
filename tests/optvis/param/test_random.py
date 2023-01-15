from __future__ import absolute_import, division, print_function

import pytest

import numpy as np
import tensorflow as tf
from lucid.optvis.param.random import image_sample


def test_image_sample():
    shape = (1, 32, 32, 3)
    with tf.compat.v1.Session() as sess:
        image_t = image_sample(shape)

        tf.compat.v1.global_variables_initializer().run()
        image1, image2 = [sess.run([image_t])[0] for _ in range(2)]
    distance = np.mean(np.absolute(image1 - image2))
    assert distance > 0
