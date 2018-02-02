# -*- coding: UTF-8 -*-
from __future__ import absolute_import, division, print_function

import pytest

import numpy as np

from lucid.util.array_to_image import _infer_image_mode_from_shape as infer_mode
from lucid.util.array_to_image import _infer_domain_from_array as infer_domain

def test_infer_image_mode_from_shape_L():
  assert infer_mode((2, 3)) == "L"
  assert infer_mode((100, 50)) == "L"
  assert infer_mode((100, 50, 1)) == "L"

def test_infer_image_mode_from_shape_RGB():
  assert infer_mode((100, 50, 3)) == "RGB"

def test_infer_image_mode_from_shape_RGBA():
  assert infer_mode((100, 50, 4)) == "RGBA"

def test_infer_image_mode_from_shape_batch():
  with pytest.raises(ValueError):
    infer_mode((2, 100, 50, 3))


def test_infer_domain_from_array_0_1():
  assert infer_domain([0, 0.01, 0.99]) == (0, 1)
  assert infer_domain([0, 0.01, 1.99]) != (0, 1)

def test_infer_domain_from_array_neg1_1():
  assert infer_domain([-0.5, 0.5]) == (-1, 1)
  assert infer_domain([-2.5, 2.5]) != (-1, 1)

def test_infer_domain_from_array_neg1_0():
  assert infer_domain([-0.5, 0]) == (-1, 0)
  assert infer_domain([-2.5, 0]) != (-1, 0)

def test_infer_domain_from_array_0_255():
  assert infer_domain([0, 127]) == (0, 255)
  assert infer_domain([-0.5, 127]) != (0, 255)

def test_normalize_array_and_convert_to_image():
  pass
