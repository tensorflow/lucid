import numpy as np
from scipy import ndimage


def resize(image, target_size, **kwargs):
    """Resize an ndarray image of rank 3 or 4.
    target_size can be a tuple `(height, width)` or scalar `width`."""

    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    if not isinstance(target_size, (list, tuple, np.ndarray)):
        message = (
            f"`target_size` should be a single number (width) or a list"
            f"/tuple/ndarray (height, width), not {type(target_size)}."
        )
        raise ValueError(message)

    rank = len(image.shape)
    assert 3 <= rank <= 4

    original_size = image.shape[-3:-1]

    if original_size == target_size:
        return image  # noop return because ndimage.zoom doesn't check itself

    # TODO: maybe allow -1 in target_size to signify aspect-ratio preserving resize?
    ratios = [t / o for t, o in zip(target_size, original_size)]
    zoom = [1] * rank
    zoom[-3:-1] = ratios

    resized = ndimage.zoom(image, zoom, **kwargs)
    assert resized.shape[-3:-1] == target_size

    return resized


def composite(
    background_image,
    foreground_image,
    foreground_width_ratio=0.25,
    foreground_position=(0.0, 0.0),
):
    """Takes two images and composites them."""

    if foreground_width_ratio <= 0:
        return background_image

    composite = background_image.copy()
    width = int(foreground_width_ratio * background_image.shape[1])
    foreground_resized = resize(foreground_image, width)
    size = foreground_resized.shape

    x = int(foreground_position[1] * (background_image.shape[1] - size[1]))
    y = int(foreground_position[0] * (background_image.shape[0] - size[0]))

    # TODO: warn if resulting coordinates are out of bounds?
    composite[y : y + size[0], x : x + size[1]] = foreground_resized

    return composite
