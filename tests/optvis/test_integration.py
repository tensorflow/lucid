from __future__ import absolute_import, division, print_function

import pytest

import tensorflow as tf
from lucid.modelzoo.vision_models import InceptionV1
from lucid.optvis import objectives, param, render, transform


model = InceptionV1()
model.load_graphdef()


@pytest.mark.parametrize("decorrelate", [True, False])
@pytest.mark.parametrize("fft", [True, False])
def test_integration(decorrelate, fft):
  obj = objectives.neuron("mixed3a_pre_relu", 0)
  param_f = lambda: param.image(16, decorrelate=decorrelate, fft=fft)
  rendering = render.render_vis(model, obj, param_f=param_f, thresholds=(1,2),
    verbose=False, transforms=[])
  start_image = rendering[0]
  end_image = rendering[-1]
  objective_f = objectives.neuron("mixed3a", 177)
  param_f = lambda: param.image(64, decorrelate=decorrelate, fft=fft)
  rendering = render.render_vis(model, objective_f, param_f, verbose=False, thresholds=(0,64), use_fixed_seed=True)
  start_image, end_image = rendering

  assert (start_image != end_image).any()
