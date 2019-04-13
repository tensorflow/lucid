from __future__ import absolute_import, division, print_function

import pytest

import tensorflow as tf
import numpy as np
from lucid.optvis import objectives, param, render, transform
from lucid.optvis.objectives import wrap_objective


np.random.seed(42)

NUM_STEPS = 3


def assert_gradient_ascent(objective, model, batch=None, alpha=False, shape=None):
    with tf.Graph().as_default() as graph, tf.Session() as sess:
        shape = shape or [1, 32, 32, 3]
        t_input = param.image(shape[1], h=shape[2], batch=batch, alpha=alpha)
        if alpha:
            t_input = transform.collapse_alpha_random()(t_input)
        model.import_graph(t_input, scope="import", forget_xy_shape=True)

        def T(layer):
            if layer == "input":
                return t_input
            if layer == "labels":
                return model.labels
            return graph.get_tensor_by_name("import/%s:0" % layer)

        loss_t = objective(T)
        opt_op = tf.train.AdamOptimizer(0.1).minimize(-loss_t)
        tf.global_variables_initializer().run()
        start_value = sess.run([loss_t])
        for _ in range(NUM_STEPS):
            _ = sess.run([opt_op])
        end_value, = sess.run([loss_t])
        print(start_value, end_value)
        assert start_value < end_value


def test_neuron(inceptionv1):
    objective = objectives.neuron("mixed4a_pre_relu", 42)
    assert_gradient_ascent(objective, inceptionv1)

def test_composition():
    @wrap_objective
    def f(a):
      return lambda T: a

    a   = f(1)
    b   = f(2)
    c   = f(3)
    ab  = a - 2*b
    cab = c*(ab - 1)

    assert str(cab)  == "F(3)·((F(1) + F(2)·2·-1) + -1)"
    assert cab(None) == 3*(1 - 2*2 - 1)
    assert a.value   == 1
    assert b.value   == 2
    assert c.value   == 3
    assert ab.value  == (a.value - 2*b.value)
    assert cab.value == c.value*(ab.value - 1)


@pytest.mark.parametrize("cossim_pow", [0, 1, 2])
def test_cossim(cossim_pow):
    true_values = [1.0, 2**(0.5)/2, 0.5]
    x = np.array([1,1], dtype = np.float32)
    y = np.array([1,0], dtype = np.float32)
    T = lambda _: tf.constant(x[None, None, None, :])
    objective = objectives.direction("dummy", y, cossim_pow=cossim_pow)
    objective_t = objective(T)
    with tf.Session() as sess:
        trueval = np.dot(x,y)*(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))**cossim_pow
        print(cossim_pow, trueval)
        objective = sess.run(objective_t)
    assert abs(objective - true_values[cossim_pow]) < 1e-3


def test_channel(inceptionv1):
    objective = objectives.channel("mixed4a_pre_relu", 42)
    assert_gradient_ascent(objective, inceptionv1)


@pytest.mark.parametrize("cossim_pow", [0, 1, 2])
def test_direction(cossim_pow, inceptionv1):
    mixed_4a_depth = 508
    random_direction = np.random.random((mixed_4a_depth))
    objective = objectives.direction(
        "mixed4a_pre_relu", random_direction, cossim_pow=cossim_pow
    )
    assert_gradient_ascent(objective, inceptionv1)


def test_direction_neuron(inceptionv1):
    mixed_4a_depth = 508
    random_direction = np.random.random([mixed_4a_depth])
    objective = objectives.direction_neuron("mixed4a_pre_relu", random_direction)
    assert_gradient_ascent(objective, inceptionv1)


def test_direction_cossim(inceptionv1):
    mixed_4a_depth = 508
    random_direction = np.random.random([mixed_4a_depth]).astype(np.float32)
    objective = objectives.direction_cossim("mixed4a_pre_relu", random_direction)
    assert_gradient_ascent(objective, inceptionv1)


def test_deepdream(inceptionv1):
    objective = objectives.deepdream("mixed4a_pre_relu")
    assert_gradient_ascent(objective, inceptionv1)


def test_tv(inceptionv1):
    objective = objectives.total_variation("mixed4a_pre_relu")
    assert_gradient_ascent(objective, inceptionv1)


def test_L1(inceptionv1):
    objective = objectives.L1()  # on input by default
    assert_gradient_ascent(objective, inceptionv1)


def test_L2(inceptionv1):
    objective = objectives.L2()  # on input by default
    assert_gradient_ascent(objective, inceptionv1)


def test_blur_input_each_step(inceptionv1):
    objective = objectives.blur_input_each_step()
    assert_gradient_ascent(objective, inceptionv1)


# TODO: add test_blur_alpha_each_step
# def test_blur_alpha_each_step(inceptionv1):
#     objective = objectives.blur_alpha_each_step()
#     assert_gradient_ascent(objective, inceptionv1, alpha=True)


def test_channel_interpolate(inceptionv1):
    # TODO: should channel_interpolate fail early if batch is available?
    objective = objectives.channel_interpolate(
        "mixed4a_pre_relu", 0, "mixed4a_pre_relu", 42
    )
    assert_gradient_ascent(objective, inceptionv1, batch=5)


def test_penalize_boundary_complexity(inceptionv1):
    # TODO: is input shape really unknown at evaluation time?
    # TODO: is the sign correctly defined on this objective? It seems I need to invert it.
    objective = objectives.penalize_boundary_complexity([1, 32, 32, 3])
    assert_gradient_ascent(-1 * objective, inceptionv1)


def test_alignment(inceptionv1):
    # TODO: is the sign correctly defined on this objective? It seems I need to invert it.
    objective = objectives.alignment("mixed4a_pre_relu")
    assert_gradient_ascent(-1 * objective, inceptionv1, batch=2)


def test_diversity(inceptionv1):
    # TODO: is the sign correctly defined on this objective? It seems I need to invert it.
    objective = objectives.diversity("mixed4a_pre_relu")
    assert_gradient_ascent(-1 * objective, inceptionv1, batch=2)


def test_input_diff(inceptionv1):
    random_image = np.random.random([1, 32, 32, 3])
    objective = objectives.input_diff(random_image)
    assert_gradient_ascent(-1 * objective, inceptionv1, batch=2)


def test_class_logit(inceptionv1):
    objective = objectives.class_logit("softmax1", "kit fox")
    assert_gradient_ascent(objective, inceptionv1, shape=[1, 224, 224, 3])
