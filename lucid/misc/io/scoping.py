import os
from copy import copy
import threading
import sys
from contextlib import contextmanager

_thread_local_scopes = threading.local()


def current_io_scopes():
    ret = getattr(_thread_local_scopes, 'io_scopes', None)
    if ret is None:
        ret = []
        _thread_local_scopes.io_scopes = ret
    return ret


def set_io_scopes(scopes):
    _thread_local_scopes.io_scopes = scopes


@contextmanager
def io_scope(path):
    current_scope = current_io_scopes()
    before = copy(current_scope)
    current_scope.append(path)
    try:
        yield
    finally:
        set_io_scopes(before)


def scope_url(url, io_scopes=None):
    io_scopes = io_scopes or current_io_scopes()
    if "//" in url or url.startswith("/"):
        return url
    paths = io_scopes + [url]
    return os.path.join(*paths)
