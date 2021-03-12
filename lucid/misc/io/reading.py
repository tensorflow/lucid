# Copyright 2018 The Lucid Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Methods for read_handle bytes from arbitrary sources.

This module takes a URL, infers how to locate it,
loads the data into memory and returns it.
"""
from contextlib import contextmanager
import hashlib
import os
import re
import logging
from urllib.parse import urlparse
from urllib import request
from tensorflow.io.gfile import GFile
import tensorflow as tf
from tempfile import gettempdir
import gc
from filelock import FileLock

from lucid.misc.io.writing import write_handle
from lucid.misc.io.scoping import scope_url


# create logger with module name, e.g. lucid.misc.io.reading
log = logging.getLogger(__name__)


# Public functions


def read(url, encoding=None, cache=None, mode="rb"):
    """Read from any URL.

    Internally differentiates between URLs supported by tf.gfile, such as URLs
    with the Google Cloud Storage scheme ('gs://...') or local paths, and HTTP
    URLs. This way users don't need to know about the underlying fetch mechanism.

    Args:
        url: a URL including scheme or a local path
        mode: mode in which to open the file. defaults to binary ('rb')
        encoding: if specified, encoding that should be used to decode read data
          if mode is specified to be text ('r'), this defaults to 'utf-8'.
        cache: whether to attempt caching the resource. Defaults to True only if
          the given URL specifies a remote resource.
    Returns:
        All bytes form the specified resource, or a decoded string of those.
    """
    with read_handle(url, cache, mode=mode) as handle:
        data = handle.read()

    if encoding:
        data = data.decode(encoding)

    return data


@contextmanager
def read_handle(url, cache=None, mode="rb"):
    """Read from any URL with a file handle.

    Use this to get a handle to a file rather than eagerly load the data:

    ```
    with read_handle(url) as handle:
    result = something.load(handle)

    result.do_something()

    ```

    When program execution leaves this `with` block, the handle will be closed
    automatically.

    Args:
        url: a URL including scheme or a local path
    Returns:
        A file handle to the specified resource if it could be reached.
        The handle will be closed automatically once execution leaves this context.
    """
    url = scope_url(url)

    scheme = urlparse(url).scheme

    if cache == "purge":
        _purge_cached(url)
        cache = None

    if _is_remote(scheme) and cache is None:
        cache = True
        log.debug("Cache not specified, enabling because resource is remote.")

    if cache:
        handle = _read_and_cache(url, mode=mode)
    else:
        if scheme in ("http", "https"):
            handle = _handle_web_url(url, mode=mode)
        elif scheme in ("gs"):
            handle = _handle_gfile(url, mode=mode)
        else:
            handle = open(url, mode=mode)

    yield handle
    handle.close()


# Handlers


def _handle_gfile(url, mode="rb"):
    return GFile(url, mode)


def _handle_web_url(url, mode="r"):
    return request.urlopen(url)


# Helper Functions


def _is_remote(scheme):
    return scheme in ("http", "https", "gs")


RESERVED_PATH_CHARS = re.compile("[^a-zA-Z0-9]")
LUCID_CACHE_DIR_NAME = "lucid_cache"
MAX_FILENAME_LENGTH = 200
_LUCID_CACHE_DIR = None  # filled on first use


def local_cache_path(remote_url):
    global _LUCID_CACHE_DIR
    """Returns the path that remote_url would be cached at locally."""
    local_name = RESERVED_PATH_CHARS.sub("_", remote_url)
    if len(local_name) > MAX_FILENAME_LENGTH:
        filename_hash = hashlib.sha256(local_name.encode("utf-8")).hexdigest()
        truncated_name = (
            local_name[: (MAX_FILENAME_LENGTH - (len(filename_hash)) - 1)]
            + "-"
            + filename_hash
        )
        log.debug(
            f"truncated long cache filename to {truncated_name} (original {len(local_name)} char name: {local_name}"
        )
        local_name = truncated_name
    if _LUCID_CACHE_DIR is None:
        _LUCID_CACHE_DIR = os.path.join(gettempdir(), LUCID_CACHE_DIR_NAME)
        if not os.path.exists(_LUCID_CACHE_DIR):
            # folder might exist if another thread/process creates it concurrently, this would be ok
            os.makedirs(_LUCID_CACHE_DIR, exist_ok=True)
            log.info(f"created lucid cache dir at {_LUCID_CACHE_DIR}")
    return os.path.join(_LUCID_CACHE_DIR, local_name)


def _purge_cached(url):
    local_path = local_cache_path(url)
    if not os.path.exists(local_path):
        return  # avoids obtaining lock if no work to do anyway
    lock = FileLock(local_path + ".lockfile")
    with lock:
        try:
            os.remove(local_path)
        except OSError:
            pass


def _read_and_cache(url, mode="rb"):
    local_path = local_cache_path(url)
    lock = FileLock(local_path + ".lockfile")
    with lock:
        if os.path.exists(local_path):
            log.debug("Found cached file '%s'.", local_path)
            return _handle_gfile(local_path)
        log.debug("Caching URL '%s' locally at '%s'.", url, local_path)
        try:
            with write_handle(local_path, "wb") as output_handle, read_handle(
                url, cache=False, mode="rb"
            ) as input_handle:
                for chunk in _file_chunk_iterator(input_handle):
                    output_handle.write(chunk)
            gc.collect()
            return _handle_gfile(local_path, mode=mode)
        except tf.errors.NotFoundError:
            raise
        except Exception as e:  # bare except to catch things like SystemExit or KeyboardInterrupt
            log.warning("Caching (%s -> %s) failed: %s", url, local_path, e)
            try:
                os.remove(local_path)
            except OSError:
                pass
            raise


from functools import partial

_READ_BUFFER_SIZE = 1048576  # setting a larger value here to help read bigger chunks of files over the network (eg from GCS)


def _file_chunk_iterator(file_handle):
    reader = partial(file_handle.read, _READ_BUFFER_SIZE)
    file_iterator = iter(reader, bytes())
    # TODO: once dropping Python <3.3 compat, update to `yield from ...`
    for chunk in file_iterator:
        yield chunk
