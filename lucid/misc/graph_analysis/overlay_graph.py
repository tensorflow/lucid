""" Simplified "overlays" on top of TensorFlow graphs.

TensorFlow graphs are often too low-level to represent our conceptual
understanding of a model. This module provides abstractions to represent
simplified graphs on top of a TensorFlow graph:

`OverlayGraph` - A subgraph of a TensorFlow computational graph. Each node
    corresponds to a node in the original TensorFlow graph, and edges
    correspond to paths through the original graph.

`OverlayNode` - A node in an OverlayGraph. Corresponds to a node in a
    TensorFlow graph.

# Example usage:

```
with tf.Graph().as_default() as graph:
  model = models.InceptionV1()
  tf.import_graph_def(model.graph_def)
  overlay = OverlayGraph(graph)
```

"""
from collections import defaultdict
import numpy as np
import tensorflow as tf


class OverlayNode():
  """A node in an OverlayGraph. Corresponds to a TensorFlow Tensor.
  """
  def __init__(self, name, overlay_graph):
    self.name = name
    self.overlay_graph = overlay_graph
    self.tf_graph = overlay_graph.tf_graph
    self.tf_node = self.tf_graph.get_tensor_by_name(name)

  @staticmethod
  def as_name(node):
    if isinstance(node, str):
      return node
    elif isinstance(node, OverlayNode):
      return node.name
    elif isinstance(node, tf.Tensor):
      return node.name

  def __repr__(self):
    return "<%s: %s>" % (self.name, self.op)

  @property
  def op(self):
    return self.tf_node.op.type

  @property
  def inputs(self):
    return self.overlay_graph.node_to_inputs[self]

  @property
  def consumers(self):
    return self.overlay_graph.node_to_consumers[self]


class OverlayGraph():
  """A subgraph of a TensorFlow computational graph.

  TensorFlow graphs are often too low-level to represent our conceptual
  understanding of a model

  OverlayGraph can be used to represent a simplified version of a TensorFlow
  graph. Each node corresponds to a node in the original TensorFlow graph, and
  edges correspond to paths through the original graph.
  """

  def __init__(self, tf_graph, nodes=None, no_pass_through=None):
    self.tf_graph = tf_graph

    if nodes is None:
      nodes = []
      for op in tf_graph.get_operations():
        nodes.extend([out.name for out in op.outputs])

    self.name_map = {name: OverlayNode(name, self) for name in nodes}
    self.no_pass_through = [] if no_pass_through is None else no_pass_through
    self.node_to_consumers = defaultdict(lambda: set())
    self.node_to_inputs = defaultdict(lambda: set())

    for node in nodes:
      node = self[node]
      for inp in self._get_overlay_inputs(node):
        self.node_to_inputs[node].add(inp)
        self.node_to_consumers[inp].add(node)

  @property
  def nodes(self):
    return [self[name] for name in self.name_map]

  def __getitem__(self, index):
    return self.name_map[OverlayNode.as_name(index)]

  def __contains__(self, item):
    return OverlayNode.as_name(item) in self.name_map

  def get_tf_node(self, node):
    name = OverlayNode.as_name(node)
    return self.tf_graph.get_tensor_by_name(name)

  def _get_overlay_inputs(self, node):
    node = self.get_tf_node(node)
    overlay_inps = []
    for inp in node.op.inputs:
      if inp in self:
        overlay_inps.append(self[inp])
      elif not node.name in self.no_pass_through:
        overlay_inps.extend(self._get_overlay_inputs(inp))
    return overlay_inps


  def graphviz(self, groups=None):
    """Print graphviz graph."""

    print("digraph G {")
    if groups is not None:
        for root, group in groups.items():
          print("")
          print(("  subgraph", "cluster_%s" % root.name.replace("/", "_"), "{"))
          print(("  label = \"%s\"") % (root.name))
          for node in group:
            print(("    \"%s\"") % node.name)
          print("  }")
    for node in self.nodes:
      for inp in node.inputs:
        print("  ", '"' + inp.name + '"', " -> ", '"' + (node.name) + '"')
    print("}")


  def filter(self, keep_nodes, pass_through=True):
    old_nodes = set(self.name_map.keys())
    new_nodes = set(keep_nodes)
    no_pass_through = set(self.no_pass_through)

    if not pass_through:
      no_pass_through += old_nodes - new_nodes

    keep_nodes = [node for node in self.name_map if node in keep_nodes]
    return OverlayGraph(self.tf_graph, keep_nodes, no_pass_through)
