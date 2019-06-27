from __future__ import absolute_import, division, print_function

import pytest

import tensorflow as tf
from lucid.optvis import objectives, param, render, transform
from lucid.modelzoo.vision_models import InceptionV1


@pytest.mark.slow
@pytest.mark.parametrize("decorrelate", [True, False])
@pytest.mark.parametrize("fft", [True, False])
def test_integration(decorrelate, fft):
    inceptionv1 = InceptionV1()
    obj = objectives.neuron("mixed3a_pre_relu", 0)
    param_f = lambda: param.image(16, decorrelate=decorrelate, fft=fft)
    rendering = render.render_vis(
        inceptionv1,
        obj,
        param_f=param_f,
        thresholds=(1, 2),
        verbose=False,
        transforms=[],
    )
    start_image = rendering[0]
    end_image = rendering[-1]
    objective_f = objectives.neuron("mixed3a", 177)
    param_f = lambda: param.image(64, decorrelate=decorrelate, fft=fft)
    rendering = render.render_vis(
        inceptionv1,
        objective_f,
        param_f,
        verbose=False,
        thresholds=(0, 64),
        use_fixed_seed=True,
    )
    start_image, end_image = rendering

    assert (start_image != end_image).any()

def arbitrary_channels_to_rgb(*args, **kwargs):
        """Arbitrary parametrization for testing"""
        channels = kwargs.pop('channels', None) or 10
        full_im = param.image(*args, channels=channels, **kwargs)
        r = tf.reduce_mean(full_im[...,:channels//3]**2, axis=-1)
        g = tf.reduce_mean(full_im[...,channels//3:2*channels//3]**2, axis=-1)
        b = tf.reduce_mean(full_im[...,2*channels//3:]**2, axis=-1)
        return tf.stack([r,g,b], axis=-1)

@pytest.mark.slow
def test_integration_any_channels():
    inceptionv1 = InceptionV1()
    objectives_f = [objectives.deepdream("mixed4a_pre_relu"), 
                objectives.channel("mixed4a_pre_relu", 360), 
                objectives.neuron("mixed3a", 177)]
    params_f = [lambda: param.grayscale_image_rgb(128),
                lambda: arbitrary_channels_to_rgb(128, channels=10)]
    for objective_f in objectives_f:
        for param_f in params_f:
            rendering = render.render_vis(
                inceptionv1,
                objective_f,
                param_f,
                verbose=False,
                thresholds=(0, 64),
                use_fixed_seed=True,
            )
            start_image, end_image = rendering

            assert (start_image != end_image).any()