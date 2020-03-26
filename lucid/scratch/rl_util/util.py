import numpy as np
import tensorflow as tf
import scipy.ndimage as nd
import lucid.optvis.render as render
from lucid.misc.io.collapse_channels import hue_to_rgb


def zoom_to(img, width):
    n = width // img.shape[-2] + 1
    img = img.repeat(n, axis=-3).repeat(n, axis=-2)
    r = float(width) / img.shape[-2]
    zoom = [1] * (img.ndim - 3) + [r, r, 1]
    return nd.zoom(img, zoom, order=0, mode="nearest")


def get_var(model, var_name):
    with tf.Graph().as_default(), tf.Session():
        t_obses = tf.placeholder(dtype=tf.float32, shape=(None, None, None, None))
        T = render.import_model(model, t_obses, t_obses)
        return T(var_name).eval()


def get_shape(model, node_name):
    with tf.Graph().as_default():
        t_obses = tf.placeholder(dtype=tf.float32, shape=(None, None, None, None))
        T = render.import_model(model, t_obses, t_obses)
        return T(node_name).get_shape().as_list()


def concatenate_horizontally(images):
    images = np.asarray(images)
    return images.transpose((1, 0, 2, 3)).reshape(
        (1, images.shape[1], images.shape[0] * images.shape[2], images.shape[3])
    )


def channels_to_rgb(X, warp=True):
    assert (X >= 0).all()

    K = X.shape[-1]

    rgb = 0
    for i in range(K):
        ang = 360 * i / K
        color = hue_to_rgb(ang, warp=warp)
        color = color[tuple(None for _ in range(len(X.shape) - 1))]
        rgb += X[..., i, None] * color

    return rgb


def conv2d(input_, filter_):
    assert input_.ndim == 4, (
        "input_ must have 4 dimensions "
        "corresponding to batch, height, width and channels"
    )
    assert (
        filter_.ndim == 2
    ), "filter_ must have 2 dimensions and will be applied channelwise"
    with tf.Graph().as_default(), tf.Session():
        filter_ = tf.tensordot(filter_, np.eye(input_.shape[-1]), axes=[[], []])
        return tf.nn.conv2d(
            input_, filter=filter_, strides=[1, 1, 1, 1], padding="SAME"
        ).eval()


def norm_filter(length, norm_ord=2, norm_func=lambda n: np.exp(-n), clip=True):
    arr = np.indices((length, length)) - ((length - 1) / 2)
    func1d = lambda x: norm_func(np.linalg.norm(x, ord=norm_ord))
    result = np.apply_along_axis(func1d, axis=0, arr=arr)
    if clip:
        bound = np.amax(np.amin(result, axis=0), axis=0)
        result *= np.logical_or(result >= bound, np.isclose(result, bound, atol=0))
    return result


def brightness_to_opacity(image):
    assert image.shape[-1] == 3
    brightness = np.apply_along_axis(
        lambda x: np.linalg.norm(x, ord=2), axis=-1, arr=image
    )[..., None]
    brightness = np.minimum(1, brightness)
    image = np.divide(
        image, brightness, out=np.zeros_like(image), where=(brightness != 0)
    )
    return np.concatenate([image, brightness], axis=-1)
