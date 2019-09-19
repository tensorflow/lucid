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
import json
from google.protobuf.message import DecodeError
import logging
import warnings
from collections import defaultdict
from itertools import chain

# create logger with module name, e.g. lucid.misc.io.reading
log = logging.getLogger(__name__)

from lucid.misc.io import load
from lucid.misc.io.saving import ClarityJSONEncoder


def load_text_labels(labels_path):
  return load(labels_path).splitlines()


def load_graphdef(model_url, reset_device=True):
  """Load GraphDef from a binary proto file."""
  graph_def = load(model_url)

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


def frozen_default_graph_def(input_node_names, output_node_names):
  """Return frozen and simplified graph_def of default graph."""

  sess = tf.get_default_session()
  if sess is None:
    raise RuntimeError("Default session not registered.")

  input_graph_def = tf.get_default_graph().as_graph_def()
  if len(input_graph_def.node) == 0:
    raise RuntimeError("Default graph is empty. Is it possible your model wasn't constructed or is in a different graph?")

  pruned_graph = tf.graph_util.remove_training_nodes(
      input_graph_def, protected_nodes=(output_node_names + input_node_names)
  )
  pruned_graph = tf.graph_util.extract_sub_graph(pruned_graph, output_node_names)

  # remove explicit device assignments
  for node in pruned_graph.node:
      node.device = ""

  all_variable_names = [v.op.name for v in tf.global_variables()]
  output_graph_def = tf.graph_util.convert_variables_to_constants(
      sess=sess,
      input_graph_def=pruned_graph,
      output_node_names=output_node_names,
      variable_names_whitelist=all_variable_names,
  )

  return output_graph_def


metadata_node_name = "lucid_metadata_json"

def infuse_metadata(graph_def, info):
  """Embed meta data as a string constant in a TF graph.

  This function takes info, converts it into json, and embeds
  it in graph_def as a constant op called `__lucid_metadata_json`.
  """
  temp_graph = tf.Graph()
  with temp_graph.as_default():
    tf.constant(json.dumps(info, cls=ClarityJSONEncoder), name=metadata_node_name)
  meta_node = temp_graph.as_graph_def().node[0]
  graph_def.node.extend([meta_node])


def extract_metadata(graph_def):
  """Attempt to extract meta data hidden in graph_def.

  Looks for a `__lucid_metadata_json` constant string op.
  If present, extract it's content and convert it from json to python.
  If not, returns None.
  """
  meta_matches = [n for n in graph_def.node if n.name==metadata_node_name]
  if meta_matches:
    assert len(meta_matches) == 1, "found more than 1 lucid metadata node!"
    meta_tensor = meta_matches[0].attr['value'].tensor
    return json.loads(meta_tensor.string_val[0])
  else:
    return None


# TODO: merge with pretty_graph's Graph class. Until then, only use this internally
class GraphDefHelper(object):
  """Allows constant time lookups of graphdef nodes by common properties."""

  def __init__(self, graph_def):
    self.graph_def = graph_def
    self.by_op = defaultdict(list)
    self.by_name = dict()
    self.by_input = defaultdict(list)
    for node in graph_def.node:
      self.by_op[node.op].append(node)
      assert node.name not in self.by_name  # names should be unique I guess?
      self.by_name[node.name] = node
      for input_name in node.input:
        self.by_input[input_name].append(node)


  def neighborhood(self, node, degree=4):
    """Am I really handcoding graph traversal please no"""
    assert self.by_name[node.name] == node
    already_visited = frontier = set([node.name])
    for _ in range(degree):
      neighbor_names = set()
      for node_name in frontier:
        outgoing = set(n.name for n in self.by_input[node_name])
        incoming = set(self.by_name[node_name].input)
        neighbor_names |= incoming | outgoing
      frontier = neighbor_names - already_visited
      already_visited |= neighbor_names
    return [self.by_name[name] for name in already_visited]
