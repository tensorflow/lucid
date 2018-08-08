import pytest
import tensorflow as tf


from lucid.optvis.transform.transparency import collapse_alpha

def test_collapse_alpha():
    reasonable_image_shape = (1,8,8,4)
    with tf.Session() as sess:
        image_alpha_t = tf.zeros(reasonable_image_shape)
        collapse_transform = collapse_alpha()
        image_t = collapse_transform(image_alpha_t)
        assert image_t.get_shape().as_list()[-1] == 3


def test_collapse_alpha_no_alpha():
    unreasonable_image_shape = (1,8,8,3)
    with pytest.raises(AssertionError):
        with tf.Session() as sess:
            image_alpha_t = tf.zeros(unreasonable_image_shape)
            collapse_transform = collapse_alpha()
            image_t = collapse_transform(image_alpha_t)
            assert image_t.get_shape().as_list()[-1] == 3
