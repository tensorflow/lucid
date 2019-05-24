import os
import sys
from contextlib import contextmanager

this = sys.modules[__name__]
this.io_scopes = []

@contextmanager
def io_scope(path):
    this.io_scopes.append(path)
    try:
        yield
    finally:
        this.io_scopes.pop()

def scope_url(url):
    if "//" in url:
        return url
    return os.path.join(*this.io_scopes, url)
