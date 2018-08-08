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

"""Utility functions for modelzoo models."""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from google.protobuf.message import DecodeError
import logging

# create logger with module name, e.g. lucid.misc.io.reading
log = logging.getLogger(__name__)

from lucid.misc.io.reading import read, local_cache_path


def load_text_labels(labels_path):
  return read(labels_path, encoding='utf-8').splitlines()


def load_graphdef(model_url, reset_device=True):
  """Load GraphDef from a binary proto file."""
  graphdef_string = read(model_url)
  try:
    graph_def = tf.GraphDef.FromString(graphdef_string)
  except DecodeError:
    cache_path = local_cache_path(model_url)
    log.error("Could not decode graphdef protobuf. Maybe check if you have a corrupted file cached at {}?".format(cache_path))
    raise RuntimeError('Could not load_graphdef!')

  if reset_device:
    for n in graph_def.node:
      n.device = ""

  return graph_def


def forget_xy(t):
  """Ignore sizes of dimensions (1, 2) of a 4d tensor in shape inference.

  This allows using smaller input sizes, which create an invalid graph at higher
  layers (for example because a spatial dimension becomes smaller than a conv
  filter) when we only use early parts of it.
  """
  shape = (t.shape[0], None, None, t.shape[3])
  return tf.placeholder_with_default(t, shape)
