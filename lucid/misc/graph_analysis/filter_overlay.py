""" Simplify `OverlayGraph`s.

Often, we want to extract a simplified version of a TensorFlow graph for
visualization and analysis. These simplified graphs can be represented as
an `OverlayGraph`, but we need a way to simplify and prune it down. That is
what this module provides.

Example use:

```
  overlay = OverlayGraph(graph)

  # Get rid of most random nodes we aren't intersted in.
  overlay = filter_overlay.ops_whitelist(overlay)

  # Get rid of nodes that aren't effected by the input.
  overlay = filter_overlay.is_dynamic(overlay)

  # Collapse sequences of nodes taht we want to think of as a single layer.
  overlay = filter_overlay.collapse_sequence(overlay, ["Conv2D", "Relu"])
  overlay = filter_overlay.collapse_sequence(overlay, ["MatMul", "Relu"])
  overlay = filter_overlay.collapse_sequence(overlay, ["MatMul", "Softmax"])
```

"""


standard_include_ops = ["Placeholder", "Relu", "Relu6", "Add", "Split", "Softmax", "Concat", "ConcatV2", "Conv2D", "MaxPool", "AvgPool", "MatMul", "EwZXy"] # Conv2D

def ops_whitelist(graph, include_ops=standard_include_ops):
  keep_nodes = [node.name for node in graph.nodes if node.op in include_ops]
  return graph.filter(keep_nodes)


def cut_shapes(graph):
  keep_nodes = [node.name for node in graph.nodes if node.op != "Shape"]
  return graph.filter(keep_nodes, pass_through=False)


def is_dynamic(graph):

  dynamic_nodes = []

  def recursive_walk_forward(node):
    if node.name in dynamic_nodes: return
    dynamic_nodes.append(node.name)
    for next in node.consumers:
      recursive_walk_forward(next)

  recursive_walk_forward(graph.nodes[0])
  return graph.filter(dynamic_nodes)


def collapse_sequence(graph, sequence):
  exclude_nodes = []

  for node in graph.nodes:
    remainder = sequence[:]
    matches = []
    while remainder:
      if node.op == remainder[0]:
        matches.append(node.name)
        remainder = remainder[1:]
      else:
        break

      if len(remainder):
        if len(node.consumers) != 1:
          break
        node = list(node.consumers)[0]

    if len(remainder) == 0:
      exclude_nodes += matches[:-1]

  include_nodes = [node.name for node in graph.nodes
                   if node.name not in exclude_nodes]

  return graph.filter(include_nodes)
