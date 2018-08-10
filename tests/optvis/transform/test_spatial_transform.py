import tensorflow as tf
import numpy as np
import math

import pytest

from lucid.optvis.transform.spatial import (
    rotate,
    scale,
    jitter,
    pad,
    homography,
    crop_or_pad_to_shape,
)
# from lucid.misc.io import save


def _find_single_dot(array):
    """Weighs entries by their value to get an approximation of where a single
    entry with value 1 in an array of 0s was mapped to.
    Asserts the point has not been spread too far (within 3px in each dim)."""

    array = np.sum(array, axis=-1)[0] # ignore batch and depth
    indices = np.where(array > 0)
    value_range = np.max(indices, axis=1) - np.min(indices, axis=1)
    assert np.all(value_range < MAX_SPREAD)

    values = array[indices]
    weighted_location = values.dot(np.transpose(indices))
    return weighted_location

MAX_SPREAD = 3 # px that a single pixel may have been stretched over
MAX_DISTANCE = 10 # distance in px that a single pixel may have been moved
MIN_DISTANCE = 0

@pytest.mark.parametrize("transform, arg", [
  (rotate, list(range(-10,10)) ),
  (scale,  [.95, 1.05]),
  (jitter, int(math.sqrt(MAX_DISTANCE))),
  (homography, None),
])
def test_spatial(transform, arg):
    image = np.zeros([1, 128, 128, 1])
    initial_location = (32, 32)
    image[0, initial_location[0], initial_location[1]] = np.ones(())
    # save(image, "image.png")
    with tf.Session() as sess:
        spatial_transform = transform(arg)
        image_t = tf.constant(image)
        transformed_t = spatial_transform(image_t)
        transformed_image = transformed_t.eval()
    # save(rotated_image, "image-rotated.png")
    new_location = _find_single_dot(transformed_image)
    distance = np.linalg.norm(initial_location - new_location)
    assert distance > MIN_DISTANCE and distance < MAX_DISTANCE


# TODO: test reflection mode, test uniform
def test_pad():
    with tf.Session() as sess:
        image_t = tf.zeros((1, 4, 6, 3))
        padding = pad(4)
        padded_image_t = padding(image_t)
        assert padded_image_t.get_shape().as_list()[1:3] == [4 + 2 * 4, 6 + 2 * 4]


@pytest.mark.parametrize(
    "target_shape",
    [
        (128, 128),  # unchanged
        (224, 224),  # padded
        (224, 196),  # asym. padded
        (64, 64),  # crop
        (64, 32),  # asym. crop
    ],
    ids=lambda t: str(t).replace(" ", ""),
)
@pytest.mark.parametrize(
    "image_shape",
    [(128, 128, 3), (1, 128, 128, 3), (2, 128, 128, 4)],
    ids=lambda t: str(t).replace(" ", ""),
)
def test_crop_or_pad_to_shape(target_shape, image_shape):
    with tf.Session() as sess:
        image_t = tf.zeros(image_shape)
        crop_transform = crop_or_pad_to_shape(target_shape)
        cropped_t = crop_transform(image_t)
        new_shape_t = tf.shape(cropped_t)
        new_shape = new_shape_t.eval()
    assert new_shape[-3] == target_shape[0]
    assert new_shape[-2] == target_shape[1]
