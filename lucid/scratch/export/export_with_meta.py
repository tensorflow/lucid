import numpy as np
import tensorflow as tf
import json

import lucid.modelzoo.vision_models as models
from lucid.misc.io import show, save, load
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform


def add_meta_node(graph_def, info):
  """Embed meta data as a string constant in a TF graph.

  This function takes info, converts it into json, and embeds
  it in graph_def as a constant op called `lucid_meta_json`.
  """
  temp_graph = tf.Graph()
  with temp_graph.as_default():
    tf.constant(json.dumps(info), name="lucid_meta_json")
  meta_node = temp_graph.as_graph_def().node[0]
  graph_def.node.extend([meta_node])


def get_meta_info(graph_def):
  """Attempt to extract meta data hidden in graph_def.

  Look for a `lucid_meta_json` constant string op. If present, extract it's
  content and convert it from json to python. If not, return None.
  """
  meta_matches = [n for n in graph_def.node if n.name=="lucid_meta_json"]
  if meta_matches:
    meta_tensor = meta_matches[0].attr['value'].tensor
    return json.loads(meta_tensor.string_val[0])
  else:
    return None


def get_frozen(input_node_names, output_node_names):
  """Return frozen and simplified graph_def of default graph."""

  sess = tf.get_default_session()
  input_graph_def = tf.get_default_graph().as_graph_def()

  pruned_graph = tf.graph_util.remove_training_nodes(
      input_graph_def, protected_nodes=(output_node_names + input_node_names)
  )
  pruned_graph = tf.graph_util.extract_sub_graph(pruned_graph, output_node_names)

  for node in pruned_graph.node:
      node.device = ""

  output_graph_def = tf.graph_util.convert_variables_to_constants(
      sess=sess,
      input_graph_def=pruned_graph,
      output_node_names=output_node_names,
      variable_names_whitelist=[v.op.name for v in tf.global_variables()],
  )

  return output_graph_def


def save_model(save_path, input_name, output_names, image_shape,
               image_value_range):

  meta_info = {
    "input_name" : input_name,
    "image_shape" : image_shape,
    "image_value_range": image_value_range,
  }

  graph_def = get_frozen(input_node_names=[input_name],
                         output_node_names=output_names)
  add_meta_node(graph_def, meta_info)

  #save(graph_def, save_path)
  with tf.gfile.GFile(save_path, "wb") as f:
    f.write(graph_def.SerializeToString())


def load_model(path):
  """Load a model with embeded meta data using only a path to its graphdef."""
  graph_def = load(path)
  meta_info = get_meta_info(graph_def)

  class DynamicModel(models.Model):
    model_path = path
    input_name = meta_info["input_name"]
    image_shape = meta_info["image_shape"]
    image_value_range = meta_info["image_value_range"]

  return DynamicModel()


def suggest_save_code():
  graph_def = tf.get_default_graph().as_graph_def()

  inferred_info = {
      "input_name" : None,
      "image_shape" : None,
      "output_names": None,
  }

  nodes_of_op = lambda s: [n.name for n in graph_def.node if n.op == s]
  node_by_name = lambda s: [n for n in graph_def.node if n.name == s][0]
  node_shape = lambda n:  [dim.size for dim in n.attr['shape'].shape.dim]

  potential_input_nodes = nodes_of_op("Placeholder")
  output_nodes = nodes_of_op("Softmax")

  if len(potential_input_nodes) == 1:
    input_name = potential_input_nodes[0]
    print("Infered: input_name = %s  (only Placeholder)" %  input_name)
    inferred_info["input_name"] = input_name
  else:
    print("Could not infer input_name.")

  if inferred_info["input_name"] is not None:
    input_node = node_by_name(inferred_info["input_name"])
    shape = node_shape(input_node)
    if len(shape) in [3,4]:
      if len(shape) == 4:
        shape = shape[1:]
      if -1 not in shape:
        print("Infered: image_shape = %s" %  shape)
        inferred_info["image_shape"] = shape
    if inferred_info["image_shape"] is None:
      print("Could not infer image_shape")

  if output_nodes:
    print("Infered: output_names = %s  (Softmax ops)" %  output_nodes)
    inferred_info["output_names"] = output_nodes
  else:
    print("Could not infer output_names.")

  print("")
  print("# Sanity check all inferred values before using this code!")
  print("save_model(")
  print("    save_path    = 'gs://save/model.pb', # TODO: replace")

  if inferred_info["input_name"] is not None:
    print("    input_name   = %s," % repr(inferred_info["input_name"]))
  else:
    print("    input_name   =   ,                   # TODO (eg. 'input' )")

  if inferred_info["output_names"] is not None:
    print("    output_names = %s," % repr(inferred_info["output_names"]) )
  else:
    print("    output_names = [ ],                  # TODO (eg. ['logits'] )")

  if inferred_info["image_shape"] is not None:
    print("    image_shape  = %s,"% repr(inferred_info["image_shape"]) )
  else:
    print("    image_shape  =   ,                   # TODO (eg. [224, 224, 3] )")

  print("    image_value_range =                  # TODO (eg. [0, 1], [0, 255], [-117, 138] )")
  print("  )")
