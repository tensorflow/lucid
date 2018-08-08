from warnings import warn

from lucid.optvis.transform.color import contrast, hue, saturation
from lucid.optvis.transform.gradient import normalize as normalize_gradient
from lucid.optvis.transform.spatial import pad, jitter, scale, rotate
from lucid.optvis.transform.transparency import collapse_alpha
from lucid.optvis.transform.utils import compose

standard_transforms = [
    pad(12, mode="constant", constant_value=.5),
    jitter(8),
    scale([1 + (i - 5) / 50. for i in range(11)]),
    rotate(list(range(-10, 11)) + 5 * [0]),
    jitter(4),
]


# Deprecated transform names

def collapse_alpha_random(*args, **kwargs):
    warn(
        "`collapse_alpha_random` is now called `collapse_alpha`, please use that instead.",
        DeprecationWarning
    )
    return collapse_alpha(*args, **kwargs)


def random_rotate(*args, **kwargs):
    warn(
        "`random_rotate` is now called `rotate`, please use that instead.",
        DeprecationWarning
    )
    return rotate(*args, **kwargs)


def random_scale(*args, **kwargs):
    warn(
        "`random_scale` is now called `scale`, please use that instead.",
        DeprecationWarning
    )
    return scale(*args, **kwargs)
