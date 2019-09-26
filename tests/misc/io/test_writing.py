# -*- coding: UTF-8 -*-
from __future__ import absolute_import, division, print_function

import pytest

from lucid.misc.io.writing import write, write_handle
from lucid.misc.io.scoping import io_scope
import lucid.misc.io.scoping as scoping
import os
import io


random_bytes = b"\x7f\x45\x4c\x46\x01\x01\x01\x00"


def test_write_text():
    text = u"The quick brown fox jumps over the lazy 🐕"
    path = "./tests/fixtures/string.txt"

    write(text, path, mode="w")
    content = io.open(path, "rt").read()

    assert os.path.isfile(path)
    assert content == text


def test_write_bytes():
    path = "./tests/fixtures/bytes"

    write(random_bytes, path)
    content = io.open(path, "rb").read()

    assert os.path.isfile(path)
    assert content == random_bytes


def test_write_handle_text():
    text = u"The quick brown 🦊 jumps over the lazy dog"
    path = "./tests/fixtures/string2.txt"

    with write_handle(path, mode="w") as handle:
        handle.write(text)
    content = io.open(path, "rt").read()

    assert os.path.isfile(path)
    assert content == text


def test_write_handle_binary():
    path = "./tests/fixtures/bytes"

    with write_handle(path) as handle:
        handle.write(random_bytes)
    content = io.open(path, "rb").read()

    assert os.path.isfile(path)
    assert content == random_bytes


def test_write_scope():
    target_path = "./tests/fixtures/write_scope.txt"
    if os.path.exists(target_path):
        os.remove(target_path)

    with io_scope("./tests/fixtures"):
        write("test", "write_scope.txt", mode="w")

    assert os.path.isfile(target_path)


def test_write_scope_nested():
    target_path = "./tests/fixtures/write_scope4.txt"
    if os.path.exists(target_path):
        os.remove(target_path)

    with io_scope("./tests"):
        with io_scope("fixtures"):
            write("test", "write_scope4.txt", mode="w")

    assert os.path.isfile(target_path)


def test_set_write_scopes():
    target_path = "./tests/fixtures/write_scope5.txt"
    if os.path.exists(target_path):
        os.remove(target_path)

    # fake write scopes, such as when called on a remote worker:
    _old_scopes = scoping.current_io_scopes()
    scoping.set_io_scopes(["./tests", "fixtures"])

    write("test", "write_scope5.txt", mode="w")

    assert os.path.isfile(target_path)

    # restore write scopes for later tests
    scoping.set_io_scopes(_old_scopes)
