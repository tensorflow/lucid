import numpy as np
import tensorflow as tf

import lucid.modelzoo.vision_models as models
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
from lucid.misc.io import show, load
from lucid.misc.io.showing import _image_url, _display_html
from collections import defaultdict


class Node(object):

  def __init__(self, name, op, graph, pretty_name=None):
    self.name = name
    self.op = op
    self.graph = graph
    self.pretty_name = pretty_name

  def __repr__(self):
    return "<%s: %s>" % (self.name, self.op)

  @property
  def inputs(self):
    return self.graph.node_to_inputs[self.name]

  @property
  def consumers(self):
    return self.graph.node_to_consumers[self.name]

  def copy(self):
    return Node(self.name, self.op, self.graph)



class Graph(object):

  def __init__(self):
    self.nodes = []
    self.name_map = {}
    self.node_to_consumers = defaultdict(lambda: [])
    self.node_to_inputs = defaultdict(lambda: [])

  def add_node(self, node):
    self.nodes.append(node)
    self.name_map[node.name] = node

  def add_edge(self, node1, node2):
    node1, node2 = self[node1], self[node2]
    self.node_to_consumers[node1.name].append(node2)
    self.node_to_inputs[node2.name].append(node1)

  def __getitem__(self, index):
    if isinstance(index, str):
      return self.name_map[index]
    elif isinstance(index, Node):
      return self.name_map[index.name]
    else:
      raise Exception("Unsupported index for Graph", type(index) )

  def graphviz(self, groups=None):
    print("digraph G {")
    if groups is not None:
        for root, group in groups.items():
          print("")
          print(("  subgraph", "cluster_%s" % root.name.replace("/", "_"), "{"))
          print(("  label = \"%s\"") % (root.pretty_name or root.name))
          for node in group:
            print(("    \"%s\"") % (node.pretty_name or node.name))
          print("  }")
    for node in self.nodes:
      for inp in node.inputs:
        print(("  ", '"' + (inp.pretty_name or inp.name) + '"', " -> ", '"' + (node.pretty_name or node.name) + '"'))
    print("}")

  @staticmethod
  def from_graphdef(graphdef):

    graph = Graph()

    for raw_node in graphdef.node:
      graph.add_node(Node(raw_node.name, raw_node.op, graph))

    for raw_node in graphdef.node:
      for raw_inp in raw_node.input:
        if raw_inp.startswith('^'):  # skip control inputs
          continue
        raw_inp_name = raw_inp.split(":")[0]
        graph.add_edge(raw_inp_name, raw_node.name)

    return graph


def filter_graph(graph, keep_nodes, pass_through=True):

  new_graph = Graph()

  for node in graph.nodes:
    if node.name in keep_nodes:
      new_node = node.copy()
      new_node.graph = new_graph
      new_node.subsumed = []
      new_graph.add_node(new_node)

  def kept_inputs(node):
    ret = []
    visited = []

    def walk(inp):
      if inp in visited: return
      visited.append(inp)
      if inp.name in keep_nodes:
        ret.append(inp)
      else:
        if pass_through:
          new_graph[node].subsumed.append(inp.name)
          for inp2 in inp.inputs:
            walk(inp2)

    for inp in node.inputs:
      walk(inp)

    return ret

  for node in graph.nodes:
    if node.name in keep_nodes:
      for inp in kept_inputs(node):
        new_graph.add_edge(inp, node)

  return new_graph


standard_include_ops = ["Placeholder", "Relu", "Relu6", "Add", "Split", "Softmax", "Concat", "ConcatV2", "Conv2D", "MaxPool", "AvgPool", "MatMul"] # Conv2D


def filter_graph_ops(graph, include_ops=standard_include_ops):
  keep_nodes = [node.name for node in graph.nodes if node.op in include_ops]
  return filter_graph(graph, keep_nodes)


def filter_graph_cut_shapes(graph):
  keep_nodes = [node.name for node in graph.nodes if node.op != "Shape"]
  return filter_graph(graph, keep_nodes, pass_through=False)


def filter_graph_dynamic(graph):

  dynamic_nodes = []

  def recursive_walk_forward(node):
    if node.name in dynamic_nodes: return
    dynamic_nodes.append(node.name)
    for next in node.consumers:
      recursive_walk_forward(next)

  recursive_walk_forward(graph.nodes[0])
  return filter_graph(graph, dynamic_nodes)


def filter_graph_collapse_sequence(graph, sequence):
  exclude_nodes = []

  for node in graph.nodes:
    remainder = sequence[:]
    matches = []
    while remainder:
      if len(node.consumers) > 1 and len(remainder) > 1:
        break
      if node.op == remainder[0]:
        matches.append(node.name)
        node = node.consumers[0]
        remainder = remainder[1:]
      else:
        break
    if len(remainder) == 0:
      exclude_nodes += matches[:-1]

  include_nodes = [node.name for node in graph.nodes
                   if node.name not in exclude_nodes]

  return filter_graph(graph, include_nodes)


def clip_node_names(graph, prefix):

  new_graph = Graph()
  for node in graph.nodes:
    new_node = node.copy()
    new_node.graph = new_graph
    new_node.subsumed = []
    new_graph.add_node(new_node)
    for inp in node.inputs:
      new_graph.add_edge(inp, new_node)

  for node in new_graph.nodes:
    if node.name.startswith(prefix):
      node.pretty_name = node.name[len(prefix):]

  return new_graph


def find_groups(graph):

  node_successors = {}
  for node in graph.nodes:
    node_successors[node.name] = set(node.inputs)
    for inp in node.inputs:
      node_successors[node.name] |= node_successors[inp.name]

  concat_nodes = [node for node in graph.nodes
                  if node.op in ["Concat", "ConcatV2", "Add"] and len(node.inputs) > 1]

  groups = {}
  group_children = set()
  for root_node in concat_nodes:
    branch_heads = root_node.inputs
    branch_nodes = [set([node]) | node_successors[node.name] for node in branch_heads]
    branch_shared = set.intersection(*branch_nodes)
    branch_uniq = set.union(*branch_nodes) - branch_shared
    groups[root_node] = set([root_node]) | branch_uniq
    group_children |= branch_uniq

  for root in list(groups.keys()):
    if root in group_children:
      del groups[root]

  return groups
