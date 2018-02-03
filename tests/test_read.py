# -*- coding: UTF-8 -*-

from __future__ import absolute_import, division, print_function

import pytest

from lucid.util.read import read, reading
import os.path
import io

path = "./tests/fixtures/string.txt"
string = u"The quick brown fox jumps over the lazy 🐕"
io.open(path, 'w').write(string)


def test_read_txt_file():
  content = read(path, mode='r')
  assert content == string


def test_reading_txt_file():
  with reading(path, mode='r') as handle:
    content1 = handle.read()
    handle.seek(0)
    content2 = handle.read()
  assert content1 == content2
  assert content1 == string


def test_read_binary_file():
  path = "./tests/fixtures/bytes"
  content = read(path)
  golden_content = io.open(path, 'rb').read()
  assert content == golden_content
