# -*- coding: UTF-8 -*-
from __future__ import absolute_import, division, print_function

import pytest

from lucid.util.write import write, writing
import os
import io
import random

def test_write_text():
  text = u"The quick brown fox jumps over the lazy 🐕"
  path = "./tests/fixtures/string.txt"

  write(text, path, mode='w')
  content = io.open(path, 'rt').read()

  assert os.path.isfile(path)
  assert content == text

def test_write_bytes():
  random_bytes = os.urandom(32)
  path = "./tests/fixtures/bytes"

  write(random_bytes, path)
  content = io.open(path, 'rb').read()

  assert os.path.isfile(path)
  assert content == random_bytes

def test_writing_text():
  text = u"The quick brown 🦊 jumps over the lazy dog"
  path = "./tests/fixtures/string2.txt"

  with writing(path, mode='w') as handle:
    handle.write(text)
  content = io.open(path, 'rt').read()

  assert os.path.isfile(path)
  assert content == text

def test_writing_binary():
    random_bytes = os.urandom(32)
    path = "./tests/fixtures/bytes"

    with writing(path) as handle:
      handle.write(random_bytes)
    content = io.open(path, 'rb').read()

    assert os.path.isfile(path)
    assert content == random_bytes
