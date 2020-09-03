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

"""Methods for loading arbitrary data from arbitrary sources.

This module takes a URL, infers its underlying data type and how to locate it,
loads the data into memory and returns a convenient representation.

This should support for example PNG images, JSON files, npy files, etc.
"""
import io
import lzma
import os
import json
import logging
import pickle

import numpy as np
import PIL.Image
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.protobuf.message import DecodeError

from lucid.misc.io.reading import read_handle
from lucid.misc.io.scoping import current_io_scopes, set_io_scopes
from lucid.misc.io.saving import nullcontext

# from lucid import modelzoo


# create logger with module name, e.g. lucid.misc.io.reading
log = logging.getLogger(__name__)


def _load_urls(urls, cache=None, **kwargs):
    if not urls:
        return []
    pages = {}
    caller_io_scopes = current_io_scopes()

    def _do_load(url):
        set_io_scopes(caller_io_scopes)
        return load(url, cache=cache, **kwargs)

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_urls = {
            executor.submit(_do_load, url): url for url in urls
        }
        for future in as_completed(future_to_urls):
            url = future_to_urls[future]
            try:
                pages[url] = future.result()
            except Exception as exc:
                pages[url] = exc
                log.error("Loading {} generated an exception: {}".format(url, exc))
    ordered = [pages[url] for url in urls]
    return ordered


def _load_npy(handle, **kwargs):
    """Load npy file as numpy array."""
    return np.load(handle, **kwargs)


def _load_img(handle, target_dtype=np.float32, size=None, **kwargs):
    """Load image file as numpy array."""

    image_pil = PIL.Image.open(handle, **kwargs)

    # resize the image to the requested size, if one was specified
    if size is not None:
        if len(size) > 2:
            size = size[:2]
            log.warning(
                "`_load_img()` received size: {}, trimming to first two dims!".format(
                    size
                )
            )
        image_pil = image_pil.resize(size, resample=PIL.Image.LANCZOS)

    image_array = np.asarray(image_pil)

    # remove alpha channel if it contains no information
    # if image_array.shape[-1] > 3 and 'A' not in image_pil.mode:
    # image_array = image_array[..., :-1]

    image_dtype = image_array.dtype
    image_max_value = np.iinfo(image_dtype).max  # ...for uint8 that's 255, etc.

    # using np.divide should avoid an extra copy compared to doing division first
    ndimage = np.divide(image_array, image_max_value, dtype=target_dtype)

    rank = len(ndimage.shape)
    if rank == 3:
        return ndimage
    elif rank == 2:
        return np.repeat(np.expand_dims(ndimage, axis=2), 3, axis=2)
    else:
        message = "Loaded image has more dimensions than expected: {}".format(rank)
        raise NotImplementedError(message)


def _load_json(handle, **kwargs):
    """Load json file as python object."""
    return json.load(handle, **kwargs)


def _load_text(handle, split=False, encoding="utf-8"):
    """Load and decode a string."""
    string = handle.read().decode(encoding)
    return string.splitlines() if split else string


def _load_graphdef_protobuf(handle, **kwargs):
    """Load GraphDef from a binary proto file."""
    # as_graph_def
    graph_def = tf.GraphDef.FromString(handle.read())

    # check if this is a lucid-saved model
    # metadata = modelzoo.util.extract_metadata(graph_def)
    # if metadata is not None:
    #   url = handle.name
    #   return modelzoo.vision_base.Model.load_from_metadata(url, metadata)

    # else return a normal graph_def
    return graph_def


def _load_pickle(handle, **kwargs):
  """Load a pickled python object."""
  return pickle.load(handle, **kwargs)


def _decompress_xz(handle, **kwargs):
    if not hasattr(handle, 'seekable') or not handle.seekable():
        # this handle is not seekable (gfile currently isn't), must load it all into memory to help lzma seek through it
        handle = io.BytesIO(handle.read())
    return lzma.LZMAFile(handle, mode="rb", format=lzma.FORMAT_XZ)


loaders = {
    ".png": _load_img,
    ".jpg": _load_img,
    ".jpeg": _load_img,
    ".npy": _load_npy,
    ".npz": _load_npy,
    ".json": _load_json,
    ".txt": _load_text,
    ".md": _load_text,
    ".pb": _load_graphdef_protobuf,
}


unsafe_loaders = {
    ".pickle": _load_pickle,
    ".pkl": _load_pickle,
}


decompressors = {
    ".xz": _decompress_xz,
}


def load(url_or_handle, allow_unsafe_formats=False, cache=None, **kwargs):
    """Load a file.

    File format is inferred from url. File retrieval strategy is inferred from
    URL. Returned object type is inferred from url extension.

    Args:
      url_or_handle: a (reachable) URL, or an already open file handle
      allow_unsafe_formats: set to True to allow saving unsafe formats (eg. pickles)
      cache: whether to attempt caching the resource. Defaults to True only if
          the given URL specifies a remote resource.

    Raises:
      RuntimeError: If file extension or URL is not supported.
    """

    # handle lists of URLs in a performant manner
    if isinstance(url_or_handle, (list, tuple)):
        return _load_urls(url_or_handle, cache=cache, **kwargs)

    ext, decompressor_ext = _get_extension(url_or_handle)
    try:
        ext = ext.lower()
        if ext in loaders:
            loader = loaders[ext]
        elif ext in unsafe_loaders:
            if not allow_unsafe_formats:
                raise ValueError(f"{ext} is considered unsafe, you must explicitly allow its use by passing allow_unsafe_formats=True")
            loader = unsafe_loaders[ext]
        else:
            raise KeyError(f'no loader found for {ext}')
        decompressor = decompressors[decompressor_ext] if decompressor_ext is not None else nullcontext
        message = "Using inferred loader '%s' due to passed file extension '%s'."
        log.debug(message, loader.__name__[6:], ext)
        return load_using_loader(url_or_handle, decompressor, loader, cache, **kwargs)
    except KeyError:
        log.warning("Unknown extension '%s', attempting to load as image.", ext)
        try:
            with read_handle(url_or_handle, cache=cache) as handle:
                result = _load_img(handle)
        except Exception as e:
            message = "Could not load resource %s as image. Supported extensions: %s"
            log.error(message, url_or_handle, list(loaders))
            raise RuntimeError(message.format(url_or_handle, list(loaders)))
        else:
            log.info("Unknown extension '%s' successfully loaded as image.", ext)
            return result


# Helpers


def load_using_loader(url_or_handle, decompressor, loader, cache, **kwargs):
    if is_handle(url_or_handle):
        with decompressor(url_or_handle) as decompressor_handle:
            result = loader(decompressor_handle, **kwargs)
    else:
        url = url_or_handle
        try:
            with read_handle(url, cache=cache) as handle:
                with decompressor(handle) as decompressor_handle:
                    result = loader(decompressor_handle, **kwargs)
        except (DecodeError, ValueError):
            log.warning(
                "While loading '%s' an error occurred. Purging cache once and trying again; if this fails we will raise an Exception! Current io scopes: %r",
                url,
                current_io_scopes(),
            )
            # since this may have been cached, it's our responsibility to try again once
            # since we use a handle here, the next DecodeError should propagate upwards
            with read_handle(url, cache="purge") as handle:
                result = load_using_loader(handle, decompressor, loader, cache, **kwargs)
    return result


def is_handle(url_or_handle):
    return hasattr(url_or_handle, "read") and hasattr(url_or_handle, "name")


def _get_extension(url_or_handle):
    compression_ext = None
    if is_handle(url_or_handle):
        path_without_ext, ext = os.path.splitext(url_or_handle.name)
    else:
        path_without_ext, ext = os.path.splitext(url_or_handle)

    if ext in decompressors:
        decompressor_ext = ext
        _, ext = os.path.splitext(path_without_ext)
    else:
        decompressor_ext = None
    if not ext:
        raise RuntimeError("No extension in URL: " + url_or_handle)
    return ext, decompressor_ext

