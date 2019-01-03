import numpy as np
import tensorflow as tf
import json

import lucid.modelzoo.vision_models as models
from lucid.misc.io.showing import _image_url, _display_html
from lucid.scratch.pretty_graphs.format_graph import complete_render_model_graph
from lucid.misc.io import show, load


def display_model(model, tooltips=None):
  tooltips = tooltips or {};
  render = complete_render_model_graph(model)

  svg = """
    <script src="https://d3js.org/d3.v5.min.js"></script>
    <style>
      .node, .background {
        fill: hsla(213, 80%%, 85%%,1);
      }
      .node:hover, .background:hover {
        fill: hsla(213, 60%%, 70%%, 1);
      }
      .background {
        opacity: 0.7;
      }
      .tooltip {
        position: absolute;
        background: #F8F8FA;
        padding: 6px;
        border-radius: 4px;
        transition: opacity 0.2s, left 0.2s, top 0.2s;
      }
    </style>

    <div style="width: %spx; height: %spx; position: relative" id="container">
    <svg width="100%%" height="100%%" style="position: absolute; top: 0px; left:0px;">

    <g transform="translate(0, 0)" class="model">
      %s
    </g>
    </svg>
    </div>
    """ % (render["shape"][0] + 200, render["shape"][1] + 40,  render["svg_inner"])
  _display_html(svg)


def display_tooltips_name(node_selector, tooltip_key_attr, container_selector="#output-body"):
  html = """
  <script>
    var nodes = d3.selectAll("{node_selector}");
    var container = d3.select("{container_selector}");
    var tooltip_div = container.append("div");
    tooltip_div.classed("tooltip", true).style("opacity", "0.0");

    function show_node() {{
      var node = d3.select(this);
      var svg_box = container.node().getBoundingClientRect();
      var box = node.node().getBoundingClientRect();
      var name = node.attr("{tooltip_key_attr}");


      tooltip_div.html(name);
      tooltip_div.style("opacity", "1.0")
        .style("left", (box.x + 4 - svg_box.x)+"px")
        .style("top", (box.y + box.height + 12 - svg_box.y)+"px");

    }}

    d3.selectAll("{node_selector}")
      .on("mouseover", show_node)
      .on("mouseout", () => tooltip_div.style("opacity", "0.0") );
    </script>
    """.format(**locals())
  #print(html)
  _display_html(html)


def display_tooltips(node_selector, tooltip_key_attr, tooltips, container_selector="#output-body"):
  tooltips_json = json.dumps(tooltips)
  html = """
  <script>
    var nodes = d3.selectAll("{node_selector}");
    var container = d3.select("{container_selector}");
    var tooltip_div = container.append("div");
    tooltip_div.classed("tooltip", true).style("opacity", "0.0");

    var tooltips = {tooltips_json};

    function show_node() {{
      var node = d3.select(this);
      var svg_box = container.node().getBoundingClientRect();
      var box = node.node().getBoundingClientRect();
      var name = node.attr("{tooltip_key_attr}");

      if (name in tooltips){{
        tooltip_div.html(tooltips[name]);
        tooltip_div.style("opacity", "1.0")
          .style("left", (box.x + 4 - svg_box.x)+"px")
          .style("top", (box.y + box.height + 12 - svg_box.y)+"px");
      }}
    }}

    d3.selectAll("{node_selector}")
      .on("mouseover", show_node)
      .on("mouseout", () => tooltip_div.style("opacity", "0.0") );
    </script>
    """.format(**locals())
  #print(html)
  _display_html(html)


def get_box(render, name):
  box_map = render["node_boxes"]
  matches = [node for node in list(box_map.keys()) if node.name == name]
  match = matches[0]
  box = box_map[match]
  return box


def p(point, dy=0):
  return "%s,%s" % (point[0], point[1] + dy)


def model_align_display(model1, model2, lines, middle_sep=150):


  render1 = complete_render_model_graph(model1)
  render2 = complete_render_model_graph(model2)
  shape1, shape2 = render1["shape"], render2["shape"]
  inner1, inner2 = render1["svg_inner"], render2["svg_inner"]
  W = max(shape1[0], shape2[0]) + 20
  H = shape1[1] + middle_sep + shape2[1] + 20

  paths = []
  for (name1, name2), weight in list(lines.items()):
    box1, box2 = get_box(render1, name1), get_box(render2, name2)
    start = [(box1[0][0]+box1[0][1])/2., shape1[1] + 15 + 10]
    end = [(box2[0][0]+box2[0][1])/2., shape1[1] + middle_sep - 10]
    d = "M %s C %s %s %s" % (p(start), p(start, dy=50), p(end, dy=-50), p(end))
    style = "stroke: hsla(20, %s%%, 70%%, %s%%); stroke-width: %spx; " % (100*(0.5 + weight/2.0), 100*weight, 3*weight)
    info = "data-tf-src=\"%s\" data-tf-dest=\"%s\" class=\"comparison-edge\" " % (name1, name2)
    path = "<path d=\"%s\" style=\"%s\" %s ></path>" % (d, style, info)
    paths.append(path)

  svg = """
    <script src="https://d3js.org/d3.v5.min.js"></script>
    <style>
      .node, .background {
        fill: hsla(213, 80%%, 85%%,1);
      }
      .node:hover, .background:hover {
        fill: hsla(213, 60%%, 70%%, 1);
      }
      .background {
        opacity: 0.7;
      }
      .comparison-edge {
        fill: none;
        stroke-linecap: round;
        transition: opacity 0.2s;
      }
      .tooltip {
        position: absolute;
        background: #F8F8FA;
        padding: 6px;
        border-radius: 4px;
        transition: opacity 0.2s, left 0.2s, top 0.2s;
      }
    </style>

    <svg width=%s height=%s>

    <g transform="translate(0, 0)" id="compare-top" class="model">
      %s
    </g>
    <g>
      %s
    </g>
    <g transform="translate(0, %s)" id="compare-bottom" class="model">
      %s
    </g>
    </svg>

    <script>
    var edges = d3.selectAll(".comparison-edge");

    function hide_edges_top() {{
      console.log("test")
      var node = d3.select(this);
      var name = node.attr("data-tf-name") || node.attr("data-group-tf-name");
      edges.each(function(d) {
        var pres = d3.select(this);
        var src = pres.attr("data-tf-src");
        pres.style("opacity", (src == name)? 1.0 : 0.0)
      })
    }}

    function hide_edges_bottom() {{
      var node = d3.select(this);
      var name = node.attr("data-tf-name") || node.attr("data-group-tf-name");
      edges.each(function(d) {
        var pres = d3.select(this);
        var src = pres.attr("data-tf-dest");
        pres.style("opacity", (src == name)? 1.0 : 0.0)
      })
    }}

    d3.selectAll("#compare-top .node, #compare-top .background")
      .on("mouseover.edges", hide_edges_top)
      .on("mouseout.edges", () => edges.style("opacity", "1.0") );

    d3.selectAll("#compare-bottom .node, #compare-bottom .background")
      .on("mouseover.edges", hide_edges_bottom)
      .on("mouseout.edges", () => edges.style("opacity", "1.0") );
    </script>
    """ % (W, H, inner1, "\n".join(paths), shape1[1] + middle_sep, inner2)

  _display_html(svg)


def vpad(space):
  _display_html("""<div style="width:20px; height: %spx"></div>""" % space)
