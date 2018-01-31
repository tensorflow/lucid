"""UnifiedIO provides wrappers around IO functions.

This is meant to make transparent which data store we are working with.
"""

from __future__ import absolute_import, division, print_function
from future.standard_library import install_aliases
install_aliases()

import os
import re
from urllib.parse import urlparse, urljoin
from future.moves.urllib.request import urlopen
from tensorflow import gfile
from tempfile import gettempdir


def read(url):
  """Read from any URL.

  Internally differentiates between URLs supported by tf.gfile, such as URLs
  with the Google Cloud Storage scheme ('gs://...') or local paths, and HTTP
  URLs. This way users don't need to know about the underlying fetch mechanism.

  Args:
    url: a URL including scheme or a local path
  Returns:
    All bytes form the specified resource if it could be reached.
  """
  scheme = urlparse(url).scheme
  if scheme in ('http', 'https'):
    return read_web_url(url)
  elif scheme == 'gs':
    return read_gcs_url(url)
  else:
    return read_path(url)


RESERVED_PATH_CHARS = re.compile("[^a-zA-Z0-9]")
def read_and_cache(url):
  local_name = RESERVED_PATH_CHARS.sub('_', url)
  local_path = os.path.join(gettempdir(), local_name)
  if os.path.exists(local_path):
    print("Trying cached file '{}'.".format(local_path))
    return read_path(local_path)
  else:
    print("Caching URL '{}' locally at '{}'.".format(url, local_path))
    result = read(url)
    save(result, local_path)
    return result


def read_web_url(url):
  print('read_web_url', url)
  return urlopen(url).read()


def read_gcs_url(url):
  # TODO: transparantly allow authenticated access through storage API
  _, resource_name = url.split('://')
  base_url = 'https://storage.googleapis.com/'
  url = urljoin(base_url, resource_name)
  return read_web_url(url)


def read_path(path):
  with gfile.Open(path, 'rb') as handle:
    result = handle.read()
  return result


def save(bytes, url):
  assert urlparse(url).scheme not in ('http', 'https')
  save_to_path(bytes, url)


def save_to_path(bytes, path):
  with gfile.Open(path, 'wb') as handle:
    handle.write(bytes)
