from __future__ import absolute_import, division, print_function

import pytest

import tensorflow as tf
from lucid.modelzoo.vision_models import InceptionV1
from lucid.optvis import objectives, param, render, transform

# model = InceptionV1()
# model.load_graphdef()
#
#
# def test_class_logit():
#   obj = objectives.neuron("mixed4c_pre_relu", 0)
#   rendering = render.render_vis(model, obj, thresholds=(1, 4), verbose=False)
#   start_image = rendering[0]
#   end_image = rendering[-1]
#   assert (start_image != end_image).any()
