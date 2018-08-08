import tensorflow as tf
import pytest

from lucid.optvis.transform.spatial import rotate, scale, jitter, pad


@pytest.mark.skip(reason="NotImplementedYet")
def test_rotate():
    pass


@pytest.mark.skip(reason="NotImplementedYet")
def test_scale():
    pass


@pytest.mark.skip(reason="NotImplementedYet")
def test_jitter():
    pass


# TODO: test reflection mode, test uniform
def test_pad():
    with tf.Session() as sess:
        image_t = tf.zeros((1,4,6,3))
        padding = pad(4)
        padded_image_t = padding(image_t)
        assert padded_image_t.get_shape().as_list()[1:3] == [4+2*4, 6+2*4]
