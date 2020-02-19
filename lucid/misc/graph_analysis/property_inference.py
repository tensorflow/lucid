"""Infer properties of TensorFlow nodes.
"""

from lucid.misc.graph_analysis.overlay_graph import OverlayNode, OverlayGraph
import tensorflow as tf

def as_tensor(t):
  if isinstance(t, OverlayNode):
    return t.tf_node
  elif isinstance(t, tf.Operation):
    return t.outputs[0]
  elif isinstance(t, tf.Tensor):
    return t


def infer_data_format(t, max_depth=20):
  """Infer data_format of a conv net activation.

  Inputs:
    t: a tf.Tensor, tf.Op, or OverlayNode

  Returns: "NHWC", "NCHW", or None
  """
  if str(t.shape) == "<unknown>" or len(t.shape) != 4:
    return None

  next_candidates = [as_tensor(t)]

  for n in range(max_depth): # 5 is random sanity limit on recursion
    inps = []
    for t in next_candidates:
      # Easiest way to find out if a tensor has an attribute seems to be trying
      try:
        return t.op.get_attr("data_format").decode("ascii")
      except:
        inps.extend(t.op.inputs)
    next_candidates = inps
  return None
