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

"""Method for saving arbitrary data to arbitrary destinations.

This module takes an object and URL, infers how to serialize and how to write
it out to the destination. The intention is to preserve your work under most
circumstances, so sometimes this will convert values by default and warn rather than
error out immediately. This sometimes means less predictable behavior.

If an object could have multiple serializations, this tries to infer the
intended serializations from the URL's file extension.

Possible extension: if not given a URL this could create one and return it?
"""
import contextlib
import logging
import lzma
import pickle
import subprocess
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor
import os.path
import json
from typing import Optional, List, Tuple
import numpy as np
import PIL.Image

from lucid.misc.io.writing import write_handle
from lucid.misc.io.serialize_array import _normalize_array
from lucid.misc.io.scoping import current_io_scopes, set_io_scopes


# create logger with module name, e.g. lucid.misc.io.saving
log = logging.getLogger(__name__)

_module_thread_locals = threading.local()


# backfill nullcontext for use before Python 3.7
if hasattr(contextlib, 'nullcontext') and False:
    nullcontext = contextlib.nullcontext
else:
    @contextlib.contextmanager
    def nullcontext(enter_result=None):
        yield enter_result


class CaptureSaveContext:
    """Keeps captured save results.
    Usage:
    save_context = CaptureSaveContext()
    with save_context:
        ...
    captured_results = save_context.captured_saves
    """

    def __init__(self):
        self.captured_saves = []

    def __enter__(self):
        if getattr(_module_thread_locals, 'save_contexts', None) is None:
            _module_thread_locals.save_contexts = []
        _module_thread_locals.save_contexts.append(self)

    def __exit__(self, exc_type, exc_value, traceback):
        _module_thread_locals.save_contexts.pop()

    def capture(self, save_result):
        if save_result is not None:
            self.captured_saves.append(save_result)

    @classmethod
    def current_save_context(cls) -> Optional['CaptureSaveContext']:
        contexts = getattr(_module_thread_locals, 'save_contexts', None)
        return contexts[-1] if contexts else None


class ClarityJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (tuple, set)):
            return list(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, "to_json"):
            return obj.to_json()
        else:
            return super(ClarityJSONEncoder, self).default(obj)


def save_json(object, handle, indent=2):
    """Save object as json on CNS."""
    obj_json = json.dumps(object, indent=indent, cls=ClarityJSONEncoder)
    handle.write(obj_json)

    return {"type": "json", "url": handle.name}


def save_npy(object, handle):
    """Save numpy array as npy file."""
    np.save(handle, object)

    return {"type": "npy", "shape": object.shape, "dtype": str(object.dtype), "url": handle.name}


def save_npz(object, handle):
    """Save dict of numpy array as npz file."""
    # there is a bug where savez doesn't actually accept a file handle.
    log.warning("Saving npz files currently only works locally. :/")
    path = handle.name
    handle.close()
    if type(object) is dict:
        np.savez(path, **object)
    elif type(object) is list:
        np.savez(path, *object)
    else:
        log.warning("Saving non dict or list as npz file, did you maybe want npy?")
        np.savez(path, object)
    return {"type": "npz", "url": path}


def save_img(object, handle, domain=None, **kwargs):
    """Save numpy array as image file on CNS."""

    if isinstance(object, np.ndarray):
        normalized = _normalize_array(object, domain=domain)
        object = PIL.Image.fromarray(normalized)

    if isinstance(object, PIL.Image.Image):
        object.save(handle, **kwargs)  # will infer format from handle's url ext.
    else:
        raise ValueError("Can only save_img for numpy arrays or PIL.Images!")

    return {
        "type": "image",
        "shape": object.size + (len(object.getbands()),),
        "url": handle.name,
    }


def save_txt(object, handle, **kwargs):
    if isinstance(object, str):
        handle.write(object)
    elif isinstance(object, list):
        for line in object:
            if isinstance(line, str):
                line = line.encode()
            if not isinstance(line, bytes):
                line_type = type(line)
                line = repr(line).encode()
                warnings.warn(
                    "`save_txt` found an object of type {}; using `repr` to convert it to string.".format(
                        line_type
                    )
                )
            if not line.endswith(b"\n"):
                line += b"\n"
            handle.write(line)

    return {"type": "txt", "url": handle.name}


def save_str(object, handle, **kwargs):
    assert isinstance(object, str)
    handle.write(object)
    return {"type": "txt", "url": handle.name}


def save_pb(object, handle, **kwargs):
    try:
        handle.write(object.SerializeToString())
    except AttributeError:
        warnings.warn(
            "`save_protobuf` failed for object {}. Re-raising original exception.".format(
                object
            )
        )
        raise
    finally:
        return {"type": "pb", "url": handle.name}


def save_pickle(object, handle, **kwargs):
  try:
    pickle.dump(object, handle)
  except AttributeError as e:
    warnings.warn("`save_pickle` failed for object {}. Re-raising original exception.".format(object))
    raise e


def compress_xz(handle, **kwargs):
    try:
        ret = lzma.LZMAFile(handle, format=lzma.FORMAT_XZ, mode="wb")
        ret.name = handle.name
        return ret
    except AttributeError as e:
        warnings.warn("`compress_xz` failed for handle {}. Re-raising original exception.".format(handle))
        raise e


savers = {
    ".png": save_img,
    ".jpg": save_img,
    ".jpeg": save_img,
    ".webp": save_img,
    ".npy": save_npy,
    ".npz": save_npz,
    ".json": save_json,
    ".txt": save_txt,
    ".pb": save_pb,
}

unsafe_savers = {
    ".pickle": save_pickle,
    ".pkl": save_pickle,
}

compressors = {
    ".xz": compress_xz,
}


def save(thing, url_or_handle, allow_unsafe_formats=False, save_context: Optional[CaptureSaveContext] = None, **kwargs):
    """Save object to file on CNS.

    File format is inferred from path. Use save_img(), save_npy(), or save_json()
    if you need to force a particular format.

    Args:
      obj: object to save.
      path: CNS path.
      allow_unsafe_formats: set to True to allow saving unsafe formats (eg. pickles)
      save_context: a context into which to capture saves, otherwise will try to use global context

    Raises:
      RuntimeError: If file extension not supported.
    """

    # Determine context
    # Is this a handle? What is the extension? Are we saving to GCS?
    is_handle = hasattr(url_or_handle, "write") and hasattr(url_or_handle, "name")
    if is_handle:
        path = url_or_handle.name
    else:
        path = url_or_handle

    path_without_ext, ext = os.path.splitext(path)
    is_gcs = path.startswith("gs://")

    if ext in compressors:
        compressor = compressors[ext]
        _, ext = os.path.splitext(path_without_ext)
    else:
        compressor = nullcontext

    if not ext:
        raise RuntimeError("No extension in URL: " + path)

    # Determine which saver should be used
    if ext in savers:
        saver = savers[ext]
    elif ext in unsafe_savers:
        if not allow_unsafe_formats:
            raise ValueError(f"{ext} is considered unsafe, you must explicitly allow its use by passing allow_unsafe_formats=True")
        saver = unsafe_savers[ext]
    elif isinstance(thing, str):
        saver = save_str
    else:
        message = "Unknown extension '{}'. As a result, only strings can be saved, not {}. Supported extensions: {}"
        raise ValueError(message.format(ext, type(thing).__name__, list(savers.keys())))

    # Actually save
    if is_handle:
        handle_provider = nullcontext
    else:
        handle_provider = write_handle

    with handle_provider(url_or_handle) as handle:
        with compressor(handle) as compressed_handle:
            result = saver(thing, compressed_handle, **kwargs)

    # Set mime type on gcs if html -- usually, when one saves an html to GCS,
    # they want it to be viewsable as a website.
    if is_gcs and ext == ".html":
        subprocess.run(
            ["gsutil", "setmeta", "-h", "Content-Type: text/html; charset=utf-8", path]
        )
    if is_gcs and ext == ".json":
        subprocess.run(
            ["gsutil", "setmeta", "-h", "Content-Type: application/json", path]
        )

    # capture save if a save context is available
    save_context = save_context if save_context is not None else CaptureSaveContext.current_save_context()
    if save_context:
        log.debug(
            "capturing save: resulted in {} -> {} in save_context {}".format(
                result, path, save_context
            )
        )
        save_context.capture(result)

    if result is not None and "url" in result and result["url"].startswith("gs://"):
        result["serve"] = "https://storage.googleapis.com/{}".format(result["url"][5:])

    return result


def batch_save(save_ops: List[Tuple], num_workers: int = 16):
    caller_io_scopes = current_io_scopes()
    current_save_context = CaptureSaveContext.current_save_context()

    def _do_save(save_op_tuple: Tuple):
        set_io_scopes(caller_io_scopes)
        if len(save_op_tuple) == 2:
            return save(save_op_tuple[0], save_op_tuple[1], save_context=current_save_context)
        elif len(save_op_tuple) == 3:
            return save(save_op_tuple[0], save_op_tuple[1], save_context=current_save_context, **(save_op_tuple[2]))
        else:
            raise ValueError(f'unknown save tuple size: {len(save_op_tuple)}')

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        save_op_futures = [executor.submit(_do_save, save_op_tuple) for save_op_tuple in save_ops]
        return [future.result() for future in save_op_futures]
