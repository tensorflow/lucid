# -*- coding: UTF-8 -*-
from __future__ import absolute_import, division, print_function

import os

import pytest

import numpy as np
from lucid.misc.io.loading import load
from lucid.misc.io.scoping import io_scope
import io

test_images = [
    "./tests/fixtures/rgbeye.png",
    "./tests/fixtures/noise_uppercase.PNG",
    "./tests/fixtures/rgbeye.jpg",
    "./tests/fixtures/noise.jpeg",
    "./tests/fixtures/image.xyz",
]

def test_load_json():
  path = "./tests/fixtures/dictionary.json"
  dictionary = load(path)
  assert "key" in dictionary


def test_load_text():
  path = "./tests/fixtures/string.txt"
  string = load(path)
  assert u"🐕" in string


def test_load_multiline_text_as_list():
  path = "./tests/fixtures/multiline.txt"
  string_list = load(path, split=True)
  assert isinstance(string_list, list)
  assert all(isinstance(string, ("".__class__, u"".__class__)) for string in string_list)


def test_load_npy():
  path = "./tests/fixtures/array.npy"
  array = load(path)
  assert array.shape is not None


def test_load_npz():
  path = "./tests/fixtures/arrays.npz"
  arrays = load(path)
  assert isinstance(arrays, np.lib.npyio.NpzFile)


@pytest.mark.parametrize("path", test_images)
def test_load_image(path):
  image = load(path)
  assert image.shape is not None
  assert all(dimension > 2 for dimension in image.shape)


@pytest.mark.parametrize("path", [
  "./tests/fixtures/rgbeye.png",
  "./tests/fixtures/noise.jpeg",
])
def test_load_image_resized(path):
  image = load(path)
  assert image.shape is not None
  assert all(dimension > 2 for dimension in image.shape)


def test_load_garbage_with_unknown_extension():
  path = "./tests/fixtures/string.XYZ"
  with pytest.raises(RuntimeError):
    load(path)


def test_load_json_with_file_handle():
  path = "./tests/fixtures/dictionary.json"
  with io.open(path, 'r') as handle:
    dictionary = load(handle)
  assert "key" in dictionary


def test_load_protobuf():
  path = "./tests/fixtures/graphdef.pb"
  graphdef = load(path)
  assert "int_val: 42" in repr(graphdef)


def test_batch_load():
    image_names = [os.path.basename(image) for image in test_images]
    with io_scope('./tests/fixtures'):
        images = load(image_names)
    assert len(images) == len(test_images)
    for i in range(len(test_images)):
        assert np.allclose(load(test_images[i]), images[i])

