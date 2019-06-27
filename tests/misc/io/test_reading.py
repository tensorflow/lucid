# -*- coding: UTF-8 -*-

from __future__ import absolute_import, division, print_function

import pytest

from lucid.misc.io.reading import read, read_handle
import os.path
import io

path = "./tests/fixtures/string.txt"
string = u"The quick brown fox jumps over the lazy 🐕"
io.open(path, 'w', encoding="utf-8").write(string)


def test_read_txt_file():
  content = read(path, encoding='utf-8')
  assert content == string


def test_read_handle_txt_file():
  with read_handle(path) as handle:
    content1 = handle.read().decode('utf-8')
  assert content1 == string


def test_read_handle_behaves_like_file():
  with read_handle(path) as handle:
    content1 = handle.read()
    handle.seek(0)
    content2 = handle.read()
  assert content1 == content2


def test_read_binary_file():
  path = "./tests/fixtures/bytes"
  content = read(path)
  golden_content = io.open(path, 'rb').read()
  assert content == golden_content


def test_read_remote_url(mocker):
  path = "https://example.com/example.html"
  golden = b"42"
  mock_urlopen = mocker.patch('future.moves.urllib.request.urlopen',
    return_value=io.BytesIO(golden))

  content = read(path, cache=False)

  mock_urlopen.assert_called_once_with(path)
  assert content == golden
