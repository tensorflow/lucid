from __future__ import absolute_import, division, print_function

import pytest

import numpy as np
from lucid.recipes.image_interpolation_params import multi_interpolation_basis


def test_multi_interpolation_basis():
  basis = multi_interpolation_basis()
  assert basis is not None

# TODO: build a larger test that actually checks that when you put the resulting
# visualizations into the objectives that their values actually increase and
# respectively decrease over the 6 or so images that this is interpolating over.
