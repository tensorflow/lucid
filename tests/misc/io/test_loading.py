# -*- coding: UTF-8 -*-
from __future__ import absolute_import, division, print_function

import pytest

import numpy as np
from lucid.misc.io.loading import load
import os.path
import io


def test_load_json():
  path = "./tests/fixtures/dictionary.json"
  dictionary = load(path)
  assert "key" in dictionary


def test_load_text():
  path = "./tests/fixtures/string.txt"
  string = load(path)
  assert u"🐕" in string


def test_load_npy():
  path = "./tests/fixtures/array.npy"
  array = load(path)
  assert array.shape is not None


def test_load_npz():
  path = "./tests/fixtures/arrays.npz"
  arrays = load(path)
  assert isinstance(arrays, np.lib.npyio.NpzFile)


@pytest.mark.parametrize("path", [
  "./tests/fixtures/rgbeye.png",
  "./tests/fixtures/noise_uppercase.PNG",
  "./tests/fixtures/rgbeye.jpg",
  "./tests/fixtures/noise.jpeg",
  "./tests/fixtures/image.xyz",
])
def test_load_image(path):
  image = load(path)
  assert image.shape is not None
  assert all(dimension > 2 for dimension in image.shape)


def test_load_garbage_with_unknown_extension():
  path = "./tests/fixtures/string.XYZ"
  with pytest.raises(RuntimeError):
    image = load(path)


def test_load_json_with_file_handle():
  path = "./tests/fixtures/dictionary.json"
  with io.open(path, 'r') as handle:
    dictionary = load(handle)
  assert "key" in dictionary
