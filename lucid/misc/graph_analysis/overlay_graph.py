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
    try:
      self.tf_node = self.tf_graph.get_tensor_by_name(name)
    except:
      self.tf_node = None
    self.sub_structure = None

  @staticmethod
  def as_name(node):
    if isinstance(node, str):
      return node
    elif isinstance(node, (OverlayNode, tf.Tensor)):
      return node.name
    else:
        raise NotImplementedError

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

  @property
  def extended_inputs(self):
    return self.overlay_graph.node_to_extended_inputs[self]

  @property
  def extended_consumers(self):
    return self.overlay_graph.node_to_extended_consumers[self]

  @property
  def gcd(self):
    return self.overlay_graph.gcd(self.inputs)

  @property
  def lcm(self):
    return self.overlay_graph.lcm(self.consumers)


class OverlayStructure():
  """Represents a sub-structure of a OverlayGraph.

  Often, we want to find structures within a graph, such as branches and
  sequences, to assist with graph layout for users.

  An OverlayStructure represents such a structure. It is typically used
  in conjunction with OverlayGraph.collapse_structures() to parse a graph.
  """

  def __init__(self, structure_type, structure):
    self.structure_type = structure_type
    self.structure = structure # A dictionary
    self.children = sum([component if isinstance(component, (list, tuple)) else [component]
                       for component in structure.values()], [])

  def __contains__(self, item):
    return OverlayNode.as_name(item) in [n.name for n in self.children]


class OverlayGraph():
  """A subgraph of a TensorFlow computational graph.

  TensorFlow graphs are often too low-level to represent our conceptual
  understanding of a model

  OverlayGraph can be used to represent a simplified version of a TensorFlow
  graph. Each node corresponds to a node in the original TensorFlow graph, and
  edges correspond to paths through the original graph.
  """

  def __init__(self, tf_graph, nodes=None, no_pass_through=None, prev_overlay=None):
    self.tf_graph = tf_graph

    if nodes is None:
      nodes = []
      for op in tf_graph.get_operations():
        nodes.extend([out.name for out in op.outputs])

    self.name_map = {name: OverlayNode(name, self) for name in nodes}
    self.nodes = [self.name_map[name] for name in nodes]
    self.no_pass_through = [] if no_pass_through is None else no_pass_through
    self.node_to_consumers = defaultdict(set)
    self.node_to_inputs = defaultdict(set)
    self.prev_overlay = prev_overlay

    for node in self.nodes:
      for inp in self._get_overlay_inputs(node):
        self.node_to_inputs[node].add(inp)
        self.node_to_consumers[inp].add(node)

    self.node_to_extended_consumers = defaultdict(set)
    self.node_to_extended_inputs = defaultdict(set)

    for node in self.nodes:
      for inp in self.node_to_inputs[node]:
        self.node_to_extended_inputs[node].add(inp)
        self.node_to_extended_inputs[node].update(self.node_to_extended_inputs[inp])

    for node in self.nodes[::-1]:
      for out in self.node_to_consumers[node]:
        self.node_to_extended_consumers[node].add(out)
        self.node_to_extended_consumers[node].update(self.node_to_extended_consumers[out])

  def __getitem__(self, index):
    return self.name_map[OverlayNode.as_name(index)]

  def __contains__(self, item):
    return OverlayNode.as_name(item) in self.name_map

  def get_tf_node(self, node):
    name = OverlayNode.as_name(node)
    return self.tf_graph.get_tensor_by_name(name)

  def _get_overlay_inputs(self, node):
    if self.prev_overlay:
      raw_inps = self.prev_overlay[node].inputs
    else:
      raw_inps = self.get_tf_node(node).op.inputs

    overlay_inps = []
    for inp in raw_inps:
      if inp.name.startswith('^'):  # skip control inputs
        continue
      if inp in self:
        overlay_inps.append(self[inp])
      elif node.name not in self.no_pass_through:
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
    keep_nodes = [self[n].name for n in keep_nodes]
    old_nodes = set(self.name_map.keys())
    new_nodes = set(keep_nodes)
    no_pass_through = set(self.no_pass_through)

    if not pass_through:
      no_pass_through += old_nodes - new_nodes

    keep_nodes = [node for node in self.name_map if node in keep_nodes]
    new_overlay = OverlayGraph(self.tf_graph, keep_nodes, no_pass_through, prev_overlay=self)
    for node in new_overlay.nodes:
      node.sub_structure = self[node].sub_structure
    return new_overlay

  def gcd(self, branches):
    """Greatest common divisor (ie. input) of several nodes."""
    branches = [self[node] for node in branches]
    branch_nodes  = [set([node]) | node.extended_inputs for node in branches]
    branch_shared =  set.intersection(*branch_nodes)
    return max(branch_shared, key=lambda n: self.nodes.index(n))

  def lcm(self, branches):
    """Lowest common multiplie (ie. consumer) of several nodes."""
    branches = [self[node] for node in branches]
    branch_nodes  = [set([node]) | node.extended_consumers for node in branches]
    branch_shared =  set.intersection(*branch_nodes)
    return min(branch_shared, key=lambda n: self.nodes.index(n))

  def sorted(self, items):
    return sorted(items, key=lambda n: self.nodes.index(self[n]))

  def collapse_structures(self, structure_map):

    keep_nodes = [node.name for node in self.nodes
                  if not any(node in structure.children for structure in structure_map.values())
                    or node in structure_map]

    new_overlay = self.filter(keep_nodes)

    for node in structure_map:
      new_overlay[node].sub_structure = structure_map[node]

    return new_overlay
