from __future__ import absolute_import, division, print_function

import pytest

import numpy as np
import tensorflow as tf
from lucid.misc.io import load
from lucid.optvis.param.color import to_valid_rgb
from lucid.optvis.param.spatial import (
    pixel_image,
    fft_image,
    laplacian_pyramid_image,
    bilinearly_sampled_image,
)


@pytest.fixture
def test_image():
    return load("./tests/fixtures/dog_cat_112.jpg")


@pytest.mark.parametrize("param", [pixel_image, fft_image, laplacian_pyramid_image])
def test_param_can_fit_image(param, test_image, maxsteps=1000):
    shape = (1,) + test_image.shape
    with tf.Session() as sess:
        image_param_t = param(shape)
        image_t = to_valid_rgb(image_param_t)
        loss_t = tf.reduce_mean((image_t - test_image) ** 2)
        dist_t = tf.reduce_mean(tf.abs(image_t - test_image))

        optimizer = tf.train.AdamOptimizer(0.05)
        optimize_op = optimizer.minimize(loss_t)

        tf.global_variables_initializer().run()
        for step in range(maxsteps):
            mean_distance, _ = sess.run([dist_t, optimize_op])
            if mean_distance < 0.01:
                break
    assert mean_distance < 0.01


def test_bilinearly_sampled_image():
    h, w = 2, 3
    img = np.float32(np.arange(6).reshape(h, w, 1))
    img = img[::-1]  # flip y to match OpenGL
    tests = [
        [0, 0, 0],
        [1, 1, 4],
        [0.5, 0.5, 2],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 1.5],
        [-1.0, -1.0, 5.0],
        [w, 1, 3.0],
        [w - 0.5, h - 0.5, 2.5],
        [2 * w - 0.5, 2 * h - 0.5, 2.5],
    ]
    tests = np.float32(tests)
    uv = np.float32((tests[:, :2] + 0.5) / [w, h])  # normalize UVs
    expected = tests[:, 2:]

    with tf.Session() as sess:
        output, = sess.run([bilinearly_sampled_image(img, uv)])
        assert np.abs(output - expected).max() < 1e-8
