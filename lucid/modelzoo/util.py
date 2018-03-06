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
from lucid.misc.io.reading import read

def load_text_labels(labels_path):
  return read(labels_path, encoding='utf-8').splitlines()

def load_graphdef(model_url, reset_device=True):
  """Load GraphDef from a binary proto file."""
  graphdef_string = read(model_url)
  graph_def = tf.GraphDef.FromString(graphdef_string)
  if reset_device:
    for n in graph_def.node:
      n.device = ""
  return graph_def

def forget_xy(t):
  """Forget sizes of dimensions [1, 2] of a 4d tensor."""
  zero = tf.identity(0)
  return t[:, zero:, zero:, :]
