import os
from copy import copy
import threading
import sys
from contextlib import contextmanager

_thread_local_scopes = threading.local()


def current_io_scopes():
    ret = getattr(_thread_local_scopes, "io_scopes", None)
    if ret is None:
        ret = []
        _thread_local_scopes.io_scopes = ret
    return ret


def set_io_scopes(scopes):
    _thread_local_scopes.io_scopes = scopes


@contextmanager
def io_scope(path, replace_current_scopes=False):
    current_scope = current_io_scopes()
    before = copy(current_scope)
    if replace_current_scopes:
        set_io_scopes(path if isinstance(path, list) else [path])
    else:
        current_scope.append(path)
    try:
        yield
    finally:
        set_io_scopes(before)


def _normalize_url(url: str) -> str:
    # os.path.normpath mangles url schemes: gs://etc -> gs:/etc
    # urlparse.urljoin doesn't normalize paths
    url_scheme, sep, url_path = url.partition("://")
    # 2021-03-12 @ludwig this method is often called with paths that are not URLs.
    # thus, url_path may be empty 
    # in this case we can't call `os.path.normpath(url_path)`
    # as it "normalizes" an empty input to "." (current directory)
    normalized_path = os.path.normpath(url_path) if url_path else ""
    joined = url_scheme + sep + normalized_path
    return joined


def scope_url(url, io_scopes=None):
    io_scopes = io_scopes or current_io_scopes()
    if "//" in url or url.startswith("/"):
        return url
    paths = io_scopes + [url]
    joined = os.path.join(*paths)
    normalized = _normalize_url(joined)
    return normalized
