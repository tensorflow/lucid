import numpy as np
import tensorflow as tf
import lucid.optvis.render as render
import itertools
from lucid.misc.gradient_override import gradient_override_map


def maxpool_override():
    def MaxPoolGrad(op, grad):
        inp = op.inputs[0]
        op_args = [
            op.get_attr("ksize"),
            op.get_attr("strides"),
            op.get_attr("padding"),
        ]
        smooth_out = tf.nn.avg_pool(inp ** 2, *op_args) / (
            1e-2 + tf.nn.avg_pool(tf.abs(inp), *op_args)
        )
        inp_smooth_grad = tf.gradients(smooth_out, [inp], grad)[0]
        return inp_smooth_grad

    return {"MaxPool": MaxPoolGrad}


def get_acts(model, layer_name, obses):
    with tf.Graph().as_default(), tf.Session():
        t_obses = tf.placeholder_with_default(
            obses.astype(np.float32), (None, None, None, None)
        )
        T = render.import_model(model, t_obses, t_obses)
        t_acts = T(layer_name)
        return t_acts.eval()


def get_grad_or_attr(
    model,
    layer_name,
    prev_layer_name,
    obses,
    *,
    act_dir=None,
    act_poses=None,
    score_fn=tf.reduce_sum,
    grad_or_attr,
    override=None,
    integrate_steps=1
):
    with tf.Graph().as_default(), tf.Session(), gradient_override_map(override or {}):
        t_obses = tf.placeholder_with_default(
            obses.astype(np.float32), (None, None, None, None)
        )
        T = render.import_model(model, t_obses, t_obses)
        t_acts = T(layer_name)
        if prev_layer_name is None:
            t_acts_prev = t_obses
        else:
            t_acts_prev = T(prev_layer_name)
        if act_dir is not None:
            t_acts = act_dir[None, None, None] * t_acts
        if act_poses is not None:
            t_acts = tf.gather_nd(
                t_acts,
                tf.concat([tf.range(obses.shape[0])[..., None], act_poses], axis=-1),
            )
        t_score = score_fn(t_acts)
        t_grad = tf.gradients(t_score, [t_acts_prev])[0]
        if integrate_steps > 1:
            acts_prev = t_acts_prev.eval()
            grad = (
                sum(
                    [
                        t_grad.eval(feed_dict={t_acts_prev: acts_prev * alpha})
                        for alpha in np.linspace(0, 1, integrate_steps + 1)[1:]
                    ]
                )
                / integrate_steps
            )
        else:
            acts_prev = None
            grad = t_grad.eval()
        if grad_or_attr == "grad":
            return grad
        elif grad_or_attr == "attr":
            if acts_prev is None:
                acts_prev = t_acts_prev.eval()
            return acts_prev * grad
        else:
            raise NotImplementedError


def get_attr(model, layer_name, prev_layer_name, obses, **kwargs):
    kwargs["grad_or_attr"] = "attr"
    return get_grad_or_attr(model, layer_name, prev_layer_name, obses, **kwargs)


def get_grad(model, layer_name, obses, **kwargs):
    kwargs["grad_or_attr"] = "grad"
    return get_grad_or_attr(model, layer_name, None, obses, **kwargs)


def get_paths(acts, nmf, *, max_paths, integrate_steps):
    acts_reduced = nmf.transform(acts)
    residual = acts - nmf.inverse_transform(acts_reduced)
    combs = itertools.combinations(range(nmf.features), nmf.features // 2)
    if nmf.features % 2 == 0:
        combs = np.array([comb for comb in combs if 0 in comb])
    else:
        combs = np.array(list(combs))
    if max_paths is None:
        splits = combs
    else:
        num_splits = min((max_paths + 1) // 2, combs.shape[0])
        splits = combs[
            np.random.choice(combs.shape[0], size=num_splits, replace=False), :
        ]
    for i, split in enumerate(splits):
        indices = np.zeros(nmf.features)
        indices[split] = 1.0
        indices = indices[tuple(None for _ in range(acts_reduced.ndim - 1))]
        complements = [False, True]
        if max_paths is not None and i * 2 + 1 == max_paths:
            complements = [np.random.choice(complements)]
        for complement in complements:
            path = []
            for alpha in np.linspace(0, 1, integrate_steps + 1)[1:]:
                if complement:
                    coordinates = (1.0 - indices) * alpha ** 2 + indices * (
                        1.0 - (1.0 - alpha) ** 2
                    )
                else:
                    coordinates = indices * alpha ** 2 + (1.0 - indices) * (
                        1.0 - (1.0 - alpha) ** 2
                    )
                path.append(
                    nmf.inverse_transform(acts_reduced * coordinates) + residual * alpha
                )
            yield path


def get_multi_path_attr(
    model,
    layer_name,
    prev_layer_name,
    obses,
    prev_nmf,
    *,
    act_dir=None,
    act_poses=None,
    score_fn=tf.reduce_sum,
    override=None,
    max_paths=50,
    integrate_steps=10
):
    with tf.Graph().as_default(), tf.Session(), gradient_override_map(override or {}):
        t_obses = tf.placeholder_with_default(
            obses.astype(np.float32), (None, None, None, None)
        )
        T = render.import_model(model, t_obses, t_obses)
        t_acts = T(layer_name)
        if prev_layer_name is None:
            t_acts_prev = t_obses
        else:
            t_acts_prev = T(prev_layer_name)
        if act_dir is not None:
            t_acts = act_dir[None, None, None] * t_acts
        if act_poses is not None:
            t_acts = tf.gather_nd(
                t_acts,
                tf.concat([tf.range(obses.shape[0])[..., None], act_poses], axis=-1),
            )
        t_score = score_fn(t_acts)
        t_grad = tf.gradients(t_score, [t_acts_prev])[0]
        acts_prev = t_acts_prev.eval()
        path_acts = get_paths(
            acts_prev, prev_nmf, max_paths=max_paths, integrate_steps=integrate_steps
        )
        deltas_of_path = lambda path: np.array(
            [b - a for a, b in zip([np.zeros_like(acts_prev)] + path[:-1], path)]
        )
        grads_of_path = lambda path: np.array(
            [t_grad.eval(feed_dict={t_acts_prev: acts}) for acts in path]
        )
        path_attrs = map(
            lambda path: (deltas_of_path(path) * grads_of_path(path)).sum(axis=0),
            path_acts,
        )
        total_attr = 0
        num_paths = 0
        for attr in path_attrs:
            total_attr += attr
            num_paths += 1
        return total_attr / num_paths
