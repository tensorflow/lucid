import numpy as np
import tensorflow as tf

import lucid.modelzoo.vision_models as models
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
from lucid.misc.io import show, load
from lucid.misc.io.showing import _image_url, _display_html
from collections import defaultdict

from lucid.scratch.pretty_graphs.graph import *

class Fragment(object):

  def __init__(self, svg, shape, node, offset=None ):
    self.svg    = svg
    self.shape = shape
    self.node = node
    self.offset = [0,0] if offset is None else offset

  def render(self):
    return """<g transform="translate(%s,%s)">%s</g>""" % (self.offset[0], self.offset[1], self.svg)

  def shift(self, delta):
    self.offset[0] += delta[0]
    self.offset[1] += delta[1]

  @property
  def fragments(self):
    return [self]

  @property
  def box(self):
    x = self.offset[0]
    y = self.offset[1]
    return [[x, x+self.shape[0]], [y, y+self.shape[1]]]


class FragmentContainer(object):

  def __init__(self, children, shape):
    self.children = children
    self.shape = shape


  def shift(self, delta):
    for child in self.children:
      child.shift(delta)

  def pad(self, pad):
    for child in self.children:
      child.shift([pad[0][0], pad[1][0]])

    self.shape[0] += pad[0][0] + pad[0][1]
    self.shape[0] += pad[1][0] + pad[1][1]

  @property
  def fragments(self):
    frags = []
    for child in self.children:
      frags += child.fragments
    return frags

  @property
  def box(self):
    boxes = [child.box for child in self.children]
    x_min = min(box[0][0] for box in boxes)
    x_max = max(box[0][1] for box in boxes)
    y_min = min(box[1][0] for box in boxes)
    y_max = max(box[1][1] for box in boxes)

    return [[x_min, x_max], [y_min, y_max]]

  def show(self, show_bounds=False):

    shape = self.shape
    container = [shape[0] + 20, shape[1] + 20]
    svg = "\n  ".join(node.render() for node in self.fragments)

    if show_bounds:
      bound_rect = """<rect x=%s y=%s width=%s height=%s fill="#E55" style="opacity: 0.3"> </rect>""" % (0, 0, shape[0], shape[1])
    else:
      bound_rect = ""

    _display_html("""
      <style>.node, .background {fill: #B8D8FF; } .node:hover, .background:hover {fill: rgb(117, 172, 240);} .background {opacity: 0.7;}</style>
      <svg width=%s height=%s>
        %s
        %s
      </svg>
    """ % (container[0], container[1], bound_rect, svg))


def one_fragement(svg, shape, node=None, inner_shape=None):
  fragment = Fragment(svg, inner_shape or shape, node)
  return FragmentContainer([fragment], shape)


def alignment_func(name):
  if name == "min":
    return lambda max_size, size: 0
  elif name == "max":
    return lambda max_size, size: max_size - size
  elif name == "mid":
    return lambda max_size, size: (max_size - size) / 2.0


class LayoutComponent(object):

  def __init__(self):
    self.pad = [[0,0], [0,0]]
    self.inner_spacing = 4

  def render(self):
    return ""


class LayoutNode(LayoutComponent):

  def __init__(self, node):
    self.pad = [[0,0], [0,0]]
    self.inner_spacing = 4
    self.node = node

  @property
  def contained_nodes(self):
    return [self.node]

  def render(self):
    if self.node == None:
      inner = ""
      shape = [10, 20]
    elif self.node.op in ["Placeholder", "Softmax"]:
      inner = "<polygon points=\"0,10 5,20 15,20 20,10 15,0 5,0 0,10\" %s></polygon>"
      shape = [20,20]
#     elif self.node.op in ["Concat", "ConcatV2"]:
#       inner = ""
#       shape = [3, 20]

    elif self.node.op in ["Concat", "ConcatV2"]:
      inner = "<rect width=10 height=20 rx=2 ry=2 x=4 %s></rect>"
      shape = [14, 20]
    elif self.node.op in ["MaxPool", "AvgPool"]:
      inner = "<polygon points=\" 0,0 0,20 3,20 10,12 10,7 3,0 0,0 \" %s></polygon>"
      shape = [10,20]
    elif self.node.op in ["Add"]:
      inner = "<circle r=6 cx=5 cy=10 %s></circle>"
      shape = [12,20]
    else:
      inner = "<rect width=10 height=20 rx=2 ry=2 %s></rect>"
      shape = [10, 20]

    if "%s" in inner:
      info = """class="node" data-tf-name="%s" data-tf-op="%s" """ % (self.node.name, self.node.op)
      inner = inner % info

    inner = "<g transform=\"translate(%s, %s)\">%s</g>" % (self.pad[0][0], self.pad[1][0], inner)
    orig_shape = shape[:]
    shape[0] += self.pad[0][0] + self.pad[0][1]
    shape[1] += self.pad[1][0] + self.pad[1][1]
    return one_fragement(inner, shape, node=self.node, inner_shape=orig_shape)


class LayoutBranch(LayoutComponent):

  def __init__(self, branches):
    self.pad = [[0,0], [0,0]]
    self.inner_spacing = 4
    self.branches = branches
    self.alignment = "max"

  @property
  def contained_nodes(self):
    return sum([branch.contained_nodes for branch in self.branches], [])

  def render(self):
    rendered_nodes = [node.render() for node in self.branches]

    align_func = alignment_func(self.alignment)

    svg   = ""
    max_x = max(node.shape[0] for node in rendered_nodes)
    delta_y = 0

    for rendered_node in rendered_nodes:
      delta_x = align_func(max_x, rendered_node.shape[0])
      rendered_node.shift([delta_x, delta_y])
      delta_y += rendered_node.shape[1] + self.inner_spacing

    if len(rendered_nodes):
      delta_y -= self.inner_spacing

    new_container = FragmentContainer(rendered_nodes, [max_x, delta_y])
    new_container.pad(self.pad)

    return new_container


class LayoutSeq(LayoutComponent):

  def __init__(self, nodes):
    self.pad = [[0,0], [0,0]]
    self.inner_spacing = 4
    self.nodes = nodes
    self.alignment = "mid"

  @property
  def contained_nodes(self):
    return sum([branch.contained_nodes for branch in self.nodes], [])

  def render(self):
    rendered_nodes = [node.render() for node in self.nodes]

    align_func = alignment_func(self.alignment)

    svg   = ""
    max_y = max(node.shape[1] for node in rendered_nodes)
    delta_x = 0

    for rendered_node in rendered_nodes:
      delta_y = align_func(max_y, rendered_node.shape[1])
      rendered_node.shift([delta_x, delta_y])
      delta_x += rendered_node.shape[0] + self.inner_spacing

    if len(rendered_nodes):
      delta_x -= self.inner_spacing

    new_container = FragmentContainer(rendered_nodes, [delta_x, max_y])
    new_container.pad(self.pad)

    return new_container

def parse_graph(graph):

  node_prevs = {}
  for node in graph.nodes:
    node_prevs[node.name] = set(node.inputs)
    for inp in node.inputs:
      node_prevs[node.name] |= node_prevs[inp.name]

  node_posts = {}
  for node in reversed(graph.nodes):
    node_posts[node.name] = set(node.consumers)
    for inp in node.consumers:
      node_posts[node.name] |= node_posts[inp.name]


  def GCA(node):
    branches = node.inputs
    branch_nodes  = [set([node]) | node_prevs[node.name] for node in branches]
    branch_shared =  set.intersection(*branch_nodes)
    return max(branch_shared, key=lambda n: graph.nodes.index(n))

  def MCP(node):
    branches = node.consumers
    branch_nodes  = [set([node]) | node_posts[node.name] for node in branches]
    branch_shared =  set.intersection(*branch_nodes)
    return min(branch_shared, key=lambda n: graph.nodes.index(n))

  def sorted_nodes(nodes):
    return sorted(nodes, key = lambda n: graph.nodes.index(n))

  def nodes_between(a, b):
    return (node_prevs[b.name] - node_prevs[a.name]) | set([a,b])

  def parse_node(node):
    return LayoutNode(node)

  def fake_node():
    return LayoutNode(None)

  def parse_list(a, b):
    seq = []
    pres = b
    while pres is not a:
      prev = GCA(pres)
      seq.append(parse_node(pres))
      if prev not in pres.inputs or len(pres.inputs) > 1:
        seq.append(parse_branch(prev, pres))
      pres = prev
    seq += [parse_node(a)]
    return LayoutSeq(list(reversed(seq)))


  def parse_branch(start, stop):
    assert GCA(stop) == start

    branches = stop.inputs
    branch_nodes  = [ (set([node]) | node_prevs[node.name])
                     - (set([start]) | node_prevs[start.name])
                     for node in branches]


    assert all([len(b1 & b2) == 0
                for b1 in branch_nodes
                for b2 in branch_nodes if b1 != b2])

    ret = []
    for nodes in branch_nodes:
      nodes_seq = sorted_nodes(nodes)
      if len(nodes) > 1:
        ret.append(parse_list(nodes_seq[0], nodes_seq[-1]))
      elif len(nodes) == 1:
        ret.append(parse_node(nodes_seq[0]))
      else:
        ret.append(fake_node())

    return LayoutBranch(ret)

  return parse_list(graph.nodes[0], graph.nodes[-1])



def render_with_groups(seq, groups, bg_pad=6, pad_diff=8, pad_none=2):

  for child in seq.nodes:
    matched_groups = [root.name for root, group_set in list(groups.items())
                      if any(grandchild in group_set for grandchild in child.contained_nodes)]
    match_group = matched_groups[0] if matched_groups else None
    child.group = match_group


  for child1, child2 in zip(seq.nodes[:-1], seq.nodes[1:]):
    if child1.group != child2.group:
      child1.pad[0][1] += pad_diff if child1.group != None else pad_none
      child2.pad[0][0] += pad_diff if child2.group != None else pad_none

    elif child1.group == None:
      child1.pad[0][1] += pad_none
      child2.pad[0][0] += pad_none

  fragment_container = seq.render()
  fragments = fragment_container.fragments

  for frag in fragments:
    frag.shift([bg_pad, bg_pad])


  # Generate groups
  used = []
  group_joined_frags = []
  for root, group_nodes in reversed(list(groups.items())):
    group_frags = [frag for frag in fragments if frag.node in group_nodes]
    used += group_frags
    box = FragmentContainer(group_frags, []).box
    bg_info = "class=\"background\" data-group-tf-name=\"%s\" " % root.name
    bg_svg = """
      <g transform="translate(%s, %s)">
        <rect width=%s height=%s rx=2 ry=2 %s></rect>
      </g>""" % (box[0][0] - bg_pad, box[1][0] - bg_pad, box[0][1] - box[0][0] + 2*bg_pad, box[1][1] - box[1][0] + 2*bg_pad, bg_info)
    svgs = [frag.render() for frag in group_frags] + [bg_svg]
    svg = """<g>%s</g>""" % "\n  ".join(svgs)
    group_joined_frags.append(Fragment(svg, [], None))

  unused = [frag for frag in fragments if frag not in used]

  all_final_node_frags = unused+group_joined_frags


  # Make edges
  node_boxes = {}
  nodes = []
  for frag in fragments:
    if frag.node:
      node_boxes[frag.node] = frag.box
      nodes.append(frag.node)

  edges = []
  def p(point, dx=0):
    return "%s,%s" % (point[0]+dx, point[1])

  for node in nodes:
    for inp in node.inputs:
      box1 = node_boxes[inp]
      mid1 = [sum(box1[0] + box1[0][1:])/3., sum(box1[1])/2.]
      box2 = node_boxes[node]
      end = [sum(box2[0] + box2[0][:1])/3., sum(box2[1])/2.]


      if end[0] - mid1[0] > 50:
        mid2 = [mid1[0] + 30, end[1]]
        dx2  = -15
      if end[0] - mid1[0] > 20:
        mid2 = [mid1[0] + 20, end[1]]
        dx2  = -8
      else:
        mid2 = end[:]
        dx2 = -6

      d = "M %s C %s %s %s L %s" % (p(mid1), p(mid1, dx=6), p(mid2, dx=dx2), p(mid2), p(end))
      info = "data-source-tf-name=\"%s\" data-dest-tf-name=\"%s\" class=\"edge\" " % (inp.name, node.name)
      path = "<path d=\"%s\" style=\"stroke: #DDD; stroke-width: 1.5px; fill: none;\" %s ></path>" % (d, info)
      frag = Fragment(path, [], None)
      edges.append(frag)


  #new_container = FragmentContainer(edges + all_final_node_grags, fragment_container.shape)
  final_fragments = edges + all_final_node_frags
  inners = [frag.render() for frag in final_fragments]
  return {"svg_inner" : "\n".join(inners), "shape" : fragment_container.shape, "node_boxes" : node_boxes }


def complete_render_model_graph(model, custom_ops=None):
  graph = Graph.from_graphdef(model.graph_def)
  include_ops = standard_include_ops
  if custom_ops is not None:
    include_ops += custom_ops
  graph = filter_graph_ops(graph, include_ops=include_ops)
  graph = filter_graph_dynamic(graph)
  graph = filter_graph_collapse_sequence(graph, ["Conv2D", "Relu"])
  parsed_graph = parse_graph(graph)
  #parsed_graph.render().show()
  if "Resnet" in model.model_path:
    print((parsed_graph.alignment))
    parsed_graph.alignment = "min"

  groups = find_groups(graph)

  return render_with_groups(parsed_graph, groups)

