from __future__ import absolute_import, division, print_function

import pytest

import numpy as np
from lucid.misc.io.saving import save
import os.path
import io


dictionary = {
  "key": "value"
}
dictionary_json = """{
  "key": "value"
}"""
array1 = np.eye(10, 10)
array2 = np.dstack([np.eye(10, 10, k=i-1) for i in range(3)])


def _remove(path):
  try:
    os.remove(path)
  except OSError:
    pass

def test_save_json():
  path = "./tests/fixtures/dictionary.json"
  _remove(path)
  save(dictionary, path)
  assert os.path.isfile(path)
  content = io.open(path, 'rt').read()
  assert content == dictionary_json


def test_save_npy():
  path = "./tests/fixtures/array.npy"
  _remove(path)
  save(array1, path)
  assert os.path.isfile(path)
  re_read_array = np.load(path)
  assert np.array_equal(array1, re_read_array)


def test_save_npz_array():
  path = "./tests/fixtures/arrays.npz"
  _remove(path)
  save([array1, array2], path)
  assert os.path.isfile(path)
  re_read_arrays = np.load(path)
  assert all(arr in re_read_arrays for arr in ("arr_0", "arr_1"))
  assert np.array_equal(array1, re_read_arrays["arr_0"])
  assert np.array_equal(array2, re_read_arrays["arr_1"])


def test_save_npz_dict():
  path = "./tests/fixtures/arrays.npz"
  _remove(path)
  arrays = { "array1": array1, "array2": array2 }
  save(arrays, path)
  assert os.path.isfile(path)
  re_read_arrays = np.load(path)
  assert all(arr in re_read_arrays for arr in list(arrays))
  assert np.array_equal(arrays["array1"], re_read_arrays["array1"])


def test_save_image_png():
  path = "./tests/fixtures/rgbeye.png"
  _remove(path)
  save(array2, path)
  assert os.path.isfile(path)


def test_save_image_jpg():
  path = "./tests/fixtures/rgbeye.jpg"
  _remove(path)
  save(array2, path)
  assert os.path.isfile(path)

def test_save_named_handle():
  path = "./tests/fixtures/rgbeye.jpg"
  _remove(path)
  with io.open(path, 'wb') as handle:
    save(array2, handle)
  assert os.path.isfile(path)
