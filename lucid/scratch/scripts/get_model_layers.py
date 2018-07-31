import lucid.optvis.render as render


def predecessors(graphdef, name):
  node_map = {}
  for n in graphdef.node:
    node_map[n.name] = n
    
  seen = []
  def get_predecessors(node_name):
    if node_name not in node_map: return []
    node = node_map[node_name]
    preds = []
    for inp in node.input:
      if inp in seen: continue
      seen.append(inp)
      preds.append(inp)
      inp_preds = get_predecessors(inp)
      preds += inp_preds
    return list(set(preds))
  
  return get_predecessors(name)


def get_branch_nodes(graphdef, concat_node):
  branch_head_names = concat_node.input[:]
  branch_head_preds = [predecessors(graphdef, name) + [name] for name in branch_head_names]
  uniqs = []
  for n in range(len(branch_head_preds)):
    pres_preds = branch_head_preds[n]
    other_preds = branch_head_preds[:n] + branch_head_preds[n+1:]
    uniqs += [name for name in pres_preds
             if not any(name in others for others in other_preds)]
  return uniqs
  
  
def propose_layers(graphdef):
  concats = [node for node in graphdef.node
             if node.op in ["Concat", "ConcatV2", "Add"]]
  branch_nodes = []
  for node in concats:
    branch_nodes += get_branch_nodes(graphdef, node)
  
  layer_proposals = [node for node in graphdef.node
             if node.op in ["Relu", "Concat", "ConcatV2", "Softmax", "Add"] and node.name not in branch_nodes]

  return layer_proposals
  
  
def propose_layers_with_shapes(model):
  proposed_layers = propose_layers(model.graph_def)
  with tf.Graph().as_default(), tf.Session() as sess:
    t_input = tf.placeholder(tf.float32, [1] + model.image_shape)
    T = render.import_model(model, t_input, t_input)
    t_shapes = [tf.shape(T(node.name))[1:] for node in proposed_layers]
    shapes = sess.run(t_shapes, {t_input: np.zeros([1] + model.image_shape)})
  return zip(proposed_layers, shapes)
  
  
def print_model_layer_code(model):
  print("layers = [")
  for node, shape in propose_layers_with_shapes(model):
      layer_type = "conv" if len(shape) == 3 else "dense"
      name = node.name.encode("ascii", "replace")
      layer = {"name": name, "type": layer_type, "size": shape[-1]}
      print("  ", layer, ",")
  print(" ]")
