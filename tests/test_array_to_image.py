# -*- coding: UTF-8 -*-
from __future__ import absolute_import, division, print_function

import pytest

import numpy as np

from lucid.util.array_to_image import _infer_image_mode_from_shape as infer_mode


@pytest.mark.parametrize("shape,mode", [
  ((2, 3), "L"),
  ((100, 50), "L"),
  ((100, 50, 1), "L"),
  ((100, 50, 3), "RGB"),
  ((100, 50, 4), "RGBA"),
])
def test_infer_image_mode_from_shape_L(shape, mode):
  assert infer_mode(shape) == mode


def test_infer_image_mode_from_shape_batch():
  with pytest.raises(ValueError):
    infer_mode((2, 100, 50, 3))
