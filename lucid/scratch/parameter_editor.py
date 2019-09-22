import numpy as np


class ParameterEditor():
  """Conveniently edit the parameters of a lucid model.

  Example usage:

    model = models.InceptionV1()
    param = ParameterEditor(model.graph_def)
    # Flip weights of first channel of conv2d0
    param["conv2d0_w", :, :, :, 0] *= -1

  """

  def __init__(self, graph_def):
    self.nodes = {}
    for node in graph_def.node:
      if "value" in node.attr:
        self.nodes[str(node.name)] = node

  def __getitem__(self, key):
    name = key[0] if isinstance(key, tuple) else key
    tensor = self.nodes[name].attr["value"].tensor
    shape = [int(d.size) for d in tensor.tensor_shape.dim]
    array = np.frombuffer(tensor.tensor_content, dtype="float32").reshape(shape).copy()
    return array[key[1:]] if isinstance(key, tuple) else array

  def __setitem__(self, key, new_value):
    name = key[0] if isinstance(key, tuple) else key
    tensor = self.nodes[name].attr["value"].tensor
    node_shape = [int(d.size) for d in tensor.tensor_shape.dim]
    if isinstance(key, tuple):
      array = np.frombuffer(tensor.tensor_content, dtype="float32")
      array = array.reshape(node_shape).copy()
      array[key[1:]] = new_value
      tensor.tensor_content = array.tostring()
    else:
      assert new_value.shape == node_shape
      tensor.tensor_content = new_value.tostring()
