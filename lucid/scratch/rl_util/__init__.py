import numpy as np
import tensorflow as tf
import sys
import importlib
from lucid.modelzoo.vision_base import Model
from lucid.misc.channel_reducer import ChannelReducer
import lucid.optvis.param as param
import lucid.optvis.objectives as objectives
import lucid.optvis.render as render
import lucid.optvis.transform as transform
from lucid.misc.io import show, save
from lucid.misc.io.showing import _image_url, _display_html

try:
    import lucid.scratch.web.svelte as lucid_svelte
except NameError:
    lucid_svelte = None
from .joblib_wrapper import load_joblib, save_joblib
from .util import (
    zoom_to,
    get_var,
    get_shape,
    concatenate_horizontally,
    hue_to_rgb,
    channels_to_rgb,
    conv2d,
    norm_filter,
    brightness_to_opacity,
)
from .attribution import (
    gradient_override_map,
    maxpool_override,
    get_acts,
    get_grad_or_attr,
    get_attr,
    get_grad,
    get_paths,
    get_multi_path_attr,
)
from .nmf import argmax_nd, LayerNMF, rescale_opacity


def all_():
    return __all__


def reload(globals_dict):
    m = importlib.reload(sys.modules[__name__])
    for f in m.__all__:
        globals_dict.update({f: getattr(m, f)})


__all__ = [
    "np",
    "tf",
    "Model",
    "ChannelReducer",
    "param",
    "objectives",
    "render",
    "transform",
    "show",
    "save",
    "_image_url",
    "_display_html",
    "lucid_svelte",
    "load_joblib",
    "save_joblib",
    "zoom_to",
    "get_var",
    "get_shape",
    "concatenate_horizontally",
    "hue_to_rgb",
    "channels_to_rgb",
    "conv2d",
    "norm_filter",
    "brightness_to_opacity",
    "gradient_override_map",
    "maxpool_override",
    "get_acts",
    "get_grad_or_attr",
    "get_attr",
    "get_grad",
    "get_paths",
    "get_multi_path_attr",
    "argmax_nd",
    "LayerNMF",
    "rescale_opacity",
    "all_",
    "reload",
]
