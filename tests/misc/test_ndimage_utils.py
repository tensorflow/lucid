import pytest
import numpy as np

from lucid.misc.io import load, save

from lucid.misc.ndimage_utils import resize, composite


@pytest.fixture()
def image():
    return load("./tests/fixtures/rgbeye.png")


def test_resize(image):
    size = (3, 3)
    resized = resize(image, size)
    assert resized.shape[-3:-1] == size


def test_resize_noop(image):
    """FYI: this essentially tests the `if original_size == target_size: return image`
    part of `resize()`'s implementation. Without it, machine precision makes the
    following assert no longer true, but I thought it was an imoprtant property to have.
    """
    resized = resize(image, image.shape[-3:-1])
    assert np.all(resized == image)


@pytest.mark.parametrize("corner", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_composite(corner):
    background_shape = (10, 10, 1)
    foreground_shape = (2, 2, 1)
    black = np.zeros(background_shape)
    white = np.ones(foreground_shape)

    comp = composite(
        black, white, foreground_position=corner, foreground_width_ratio=0.2
    )

    wh = comp.shape[-3:-1]
    xy = [min(d * c, d - 1) for d, c in zip(wh, corner)]
    print(wh, corner, xy)
    assert comp[xy[0], xy[1], 0] > 0.99  # because interpolation!
    assert comp[5, 5] == 0
