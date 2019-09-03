from timeit import default_timer as timer

import tensorflow as tf
import numpy as np

from lucid.optvis import render
from lucid.optvis import param
from lucid.optvis import transform
from lucid.optvis import objectives
from lucid.optvis import overrides
from lucid.misc.io import show, load
from lucid.modelzoo.vision_base import Layer


def neuron(*args, **kwargs):
    return _main(objectives.neuron, *args, **kwargs)


def channel(*args, **kwargs):
    return _main(objectives.channel, *args, **kwargs)


def _main(
    objective_f,
    model,
    layer,
    channel_indices,
    alpha=False,
    negative=False,
    decorrelation_matrix=None,
    n_steps=128,
    parallel_transforms=16,
    lr=0.05,
    override_gradients=True,
):
    if not isinstance(channel_indices, (tuple, list)):
        channel_indices = [channel_indices]

    if isinstance(layer, Layer):
      layer_name = layer.name
    elif isinstance(layer, str):
      layer_name = layer
    else:
      raise ValueError("layer argument can be either a Layer object or str")

    sign = -1 if negative else 1
    batch = len(channel_indices)
    w, h, _ = model.image_shape

    with tf.Graph().as_default(), tf.Session() as sess:

        # Transformation Robustness
        transforms = transform.standard_transforms + [transform.crop_or_pad_to(w, h)]
        if alpha:
            transforms += [transform.collapse_alpha_random()]
        transform_f = render.make_transform_f(transforms)

        # Parameterization
        image_t = param.image(
            w, h, fft=True, decorrelate=True, alpha=alpha, batch=batch
        )
        param_t = transform_f(image_t)

        # Gradient Overrides
        if override_gradients:
            with overrides.relu_overrides():
                T = render.import_model(model, param_t, image_t)
        else:
            T = render.import_model(model, param_t, image_t)

        # Objective
        if decorrelation_matrix is None:
            objs = [
                sign * objective_f(layer_name, i, batch=b)
                for b, i in enumerate(channel_indices)
            ]
        else:
            raise NotImplementedError

        reported_obj = tf.stack([o(T) for o in objs])

        obj = sum(objs)(T)

        if alpha:
            obj *= 1.0 - tf.reduce_mean(image_t[..., -1])
            obj -= 0.1 * objectives.blur_alpha_each_step()(T)

        # Optimization

        optimization = tf.train.AdamOptimizer(lr).minimize(-obj)
        tf.global_variables_initializer().run()
        losses = np.zeros((batch, n_steps))
        for step in range(n_steps):
            _, loss = sess.run([optimization, reported_obj])
            losses[:, step] = loss

        # Evaluation
        visualization = image_t.eval()

    if batch == 1:
      return (visualization, losses)
    else:
      return list(zip(visualization, losses))


# def manifest_neuron(model, layer_name, channel_index, **kwargs):
#     start = timer()
#     visualization, losses = neuron(model, layer_name, channel_index, **kwargs)
#     end = timer()
#     elapsed = end - start

#     results = {
#         "type": "feature-visualization-neuron",
#         "took": elapsed,
#         "layer_name": layer_name,
#         "channel_index": channel_index,
#         "objective_values": losses,
#     }

#     results["values"] = [
#         {"type": "image", "value": visualization, "shape": visualization.shape}
#     ]

#     return results
