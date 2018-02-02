# Copyright 2018 The Deepviz Authors. All Rights Reserved.
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

"""Methods for reading bytes from arbitrary sources.

This module takes a URL, infers how to locate it,
loads the data into memory and returns it.
"""

from __future__ import absolute_import, division, print_function
from future.standard_library import install_aliases
install_aliases()

from contextlib import contextmanager
import os
import re
import logging
from urllib.parse import urlparse, urljoin
from future.moves.urllib.request import urlopen
from tensorflow import gfile
from tempfile import gettempdir

from lucid.util.write import write


RESERVED_PATH_CHARS = re.compile("[^a-zA-Z0-9]")


def _supports_binary_reading(url):
  return True


def read(url, mode='rb', cache=True):
  """Read from any URL.

  Internally differentiates between URLs supported by tf.gfile, such as URLs
  with the Google Cloud Storage scheme ('gs://...') or local paths, and HTTP
  URLs. This way users don't need to know about the underlying fetch mechanism.

  Args:
    url: a URL including scheme or a local path
  Returns:
    All bytes form the specified resource if it could be reached.
  """
  if cache and not is_local(url):
    return read_and_cache(url)

  scheme = urlparse(url).scheme
  if scheme in ('http', 'https'):
    return read_web_url(url)
  elif scheme == 'gs':
    return read_gcs_url(url)
  else:
    return read_path(url, mode=mode)


@contextmanager
def reading(path, mode=None):
  if mode is None:
    if _supports_binary_reading(path):
      mode = 'rb'
    else:
      mode = 'rt'

  handle = gfile.Open(path, mode)
  yield handle
  handle.close()


def is_local(url):
  scheme = urlparse(url).scheme
  print("scheme", scheme)
  return scheme not in ('http', 'https', 'gs')


def read_and_cache(url):
  local_name = RESERVED_PATH_CHARS.sub('_', url)
  local_path = os.path.join(gettempdir(), local_name)
  if os.path.exists(local_path):
    logging.info("Found cached file '%s'.", local_path)
    return read_path(local_path)
  else:
    logging.info("Caching URL '%s' locally at 's'.", url, local_path)
    result = read(url, cache=False)  # important to avoid endless loop
    write(result, local_path)
    return result


def read_web_url(url):
  logging.debug('read_web_url s', url)
  return urlopen(url).read()


def read_gcs_url(url):
  logging.debug('read_gcs_url s', url)
  # TODO: transparantly allow authenticated access through storage API
  _, resource_name = url.split('://')
  base_url = 'https://storage.googleapis.com/'
  url = urljoin(base_url, resource_name)
  return read_web_url(url)


def read_path(path, mode='rb'):
  logging.debug('read_path %s %s', path, mode)
  with gfile.Open(path, mode) as handle:
    result = handle.read()
  return result
