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

"""Methods for writing bytes to arbitrary destinations.

This module takes data and a URL, and attempts to save that data at that URL.

"""

from __future__ import absolute_import, division, print_function

from contextlib import contextmanager
from urllib.parse import urlparse
from tensorflow import gfile
import os


def _supports_make_dirs(path):
  """Whether this path implies a storage system that supports and requires
  intermediate directories to be created explicitly."""
  return not path.startswith("/bigstore")


def _supports_binary_writing(path):
  """Whether this path implies a storage system that supports and requires
  intermediate directories to be created explicitly."""
  return not path.startswith("/bigstore")


def _write_to_path(data, path, mode='wb'):
  with write_handle(path, mode) as handle:
    handle.write(data)


# Public functions


def write(data, url, mode='wb'):
  if urlparse(url).scheme in ('http', 'https'):
    message = "Writing to remote URL (%s) is not yet supported."
    raise ValueError(message, url)

  _write_to_path(data, url, mode=mode)


@contextmanager
def write_handle(path, mode=None):

  if _supports_make_dirs(path):
    gfile.MakeDirs(os.path.dirname(path))

  if mode is None:
    if _supports_binary_writing(path):
      mode = 'wb'
    else:
      mode = 'w'

  handle = gfile.Open(path, mode)
  yield handle
  handle.close()
