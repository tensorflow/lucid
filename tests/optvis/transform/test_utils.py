from lucid.optvis.transform.utils import compose, rand_select, angle2rads

import pytest
import math
import tensorflow as tf


def test_compose():
    adders = [lambda x: x+1] * 10
    ten_adder = compose(adders)
    assert ten_adder(0) == 10


def test_rand_select():
    elements = list(range(10))
    last_result = None
    for counter in range(100):
        with tf.Session() as sess:
            elements_t = tf.constant(elements)
            result_t = rand_select(elements)
            result, = sess.run([result_t])
        assert result in elements
        if last_result is not None and last_result != result:
            return # Accept test
        else:
            last_result = result
    assert False, "rand_select produced the same result 100 times."


@pytest.mark.parametrize("input,unit,output", [
    (180, "degrees", math.pi),
    (90, "degs", math.pi/2),
    (45, "deg", math.pi/4),
    (math.pi, "radians", math.pi),
    (57.295779513, "degrees", 1),
])
def test_angle2rads(input, unit, output):
    with tf.Session() as sess:
        input_t = tf.constant(input)
        result_t = angle2rads(input_t, unit)
        result, = sess.run([result_t])
    assert result == pytest.approx(output)
