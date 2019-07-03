from timeit import default_timer as timer

import tensorflow as tf

from lucid.optvis import render
from lucid.optvis import param
from lucid.optvis import transform
from lucid.optvis import objectives
from lucid.optvis import overrides
from lucid.misc.io import show, load
from lucid.misc.ndimage_utils import resize


def image_activations(model, image, layer_names=None):
    if layer_names is None:
        layer_names = [layer["name"] for layer in model.layers if "conv" in layer.tags]

    resized_image = resize(image, model.image_shape[:2])

    with tf.Graph().as_default() as graph, tf.Session() as sess:
        image_t = tf.placeholder_with_default(resized_image, shape=model.image_shape)
        model.import_graph(image_t, scope="import")
        layer_ts = {}
        for layer_name in layer_names:
          name = layer_name if layer_name.endswith(":0") else layer_name + ":0"
          layer_t = graph.get_tensor_by_name("import/{}".format(name))[0]
          layer_ts[layer_name] = layer_t
        activations = sess.run(layer_ts)

    return activations


def manifest_image_activations(model, image, **kwargs):
    start = timer()
    activations_dict = image_activations(model, image, **kwargs)
    end = timer()
    elapsed = end - start

    results = {"type": "image-activations", "took": elapsed}

    results["values"] = [
        {
            "type": "activations",
            "value": value,
            "shape": value.shape,
            "layer_name": layer_name,
        }
        for layer_name, value in activations_dict.items()
    ]

    return results
