import tensorflow as tf
import numpy as np
import pytest

from lucid.optvis.transform.color import contrast, hue, saturation, jitter
from lucid.misc.io import save

@pytest.mark.parametrize("transform, args", [
  (contrast, [.8, 1.2]), # 0.8 to 1.2 by default
  (hue, [10]), # 10 degrees rotation in color space
  (saturation, [.1]), # from 0.9 to 1.1
  (jitter, [4]), # jitter channels independently by max of this distance
])
def test_color(transform, args):
    """Smoke test that simply executes transform, no assertions."""
    image = np.random.uniform(size=[1, 128, 128, 3]).astype(np.float32)
    with tf.Session() as sess:
        color_transform = transform(*args)
        image_t = tf.constant(image)
        transformed_t = color_transform(image_t)
        transformed_image = transformed_t.eval()


@pytest.mark.skip(reason="NotImplementedYet")
def test_jitter():
    pass


@pytest.mark.skip(reason="NotImplementedYet")
def test_saturation():
    pass


@pytest.mark.skip(reason="NotImplementedYet")
def test_hue():
    pass


@pytest.mark.skip(reason="NotImplementedYet")
def test_contrast():
    pass
