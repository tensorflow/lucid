import os
from copy import copy
import sys
from contextlib import contextmanager

this = sys.modules[__name__]
this.io_scopes = []


@contextmanager
def io_scope(path):
    before = copy(this.io_scopes)
    this.io_scopes.append(path)
    try:
        yield
    finally:
        this.io_scopes = before


def scope_url(url, io_scopes=None):
    io_scopes = io_scopes or this.io_scopes
    if "//" in url or url.startswith("/"):
        return url
    return os.path.join(*io_scopes, url)
