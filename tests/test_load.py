from __future__ import absolute_import, division, print_function

import pytest

import numpy as np
from lucid.util.load import load
import os.path
import io


def test_load_json():
  path = "./tests/fixtures/dictionary.json"
  dictionary = load(path)
  assert "key" in dictionary


def test_load_npy():
  path = "./tests/fixtures/array.npy"
  array = load(path)
  assert array.shape is not None


def test_load_npz():
  path = "./tests/fixtures/arrays.npz"
  arrays = load(path)
  assert isinstance(arrays, np.lib.npyio.NpzFile)


@pytest.mark.parametrize("path", [
  "./tests/fixtures/noise.png",
  "./tests/fixtures/noise.PNG",
  "./tests/fixtures/noise.jpg",
  "./tests/fixtures/noise.jpeg",
  "./tests/fixtures/image.XYZ",
])
def test_load_image(path):
  image = load(path)
  assert image.shape is not None
  assert all(dimension > 2 for dimension in image.shape)


def test_load_garbage_with_unknown_extension():
  path = "./tests/fixtures/string.XYZ"
  with pytest.raises(RuntimeError):
    image = load(path)
