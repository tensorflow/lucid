import pytest
from lucid.misc.io.scoping import io_scope, scope_url


def test_empty_io_scope():
    path = "./some/file.ext"
    scoped = scope_url(path)
    assert scoped == path
