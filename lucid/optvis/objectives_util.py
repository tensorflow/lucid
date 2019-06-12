
from __future__ import absolute_import, division, print_function

from decorator import decorator
import numpy as np
import tensorflow as tf

import lucid.misc.graph_analysis.property_inference as property_inference


def _dot(x, y):
  return tf.reduce_sum(x * y, -1)


def _dot_cossim(x, y, cossim_pow=0):
  eps = 1e-4
  xy_dot = _dot(x, y)
  if cossim_pow == 0: return tf.reduce_mean(xy_dot)
  x_mags = tf.sqrt(_dot(x,x))
  y_mags = tf.sqrt(_dot(y,y))
  cossims = xy_dot / (eps + x_mags ) / (eps + y_mags)
  floored_cossims = tf.maximum(0.1, cossims)
  return tf.reduce_mean(xy_dot * floored_cossims**cossim_pow)


def _extract_act_pos(acts, x=None, y=None):
  shape = tf.shape(acts)
  x_ = shape[1] // 2 if x is None else x
  y_ = shape[2] // 2 if y is None else y
  return acts[:, x_:x_+1, y_:y_+1]


def _make_arg_str(arg):
  arg = str(arg)
  too_big = len(arg) > 15 or "\n" in arg
  return "..." if too_big else arg


def _T_force_NHWC(T):
  """Modify T to accomdate different data types

    [N, C, H, W]  ->  [N, H, W, C]
    [N, C]        ->  [N, 1, 1, C]

  """
  def T2(name):
    t = T(name)
    shape = t.shape
    if str(shape) == "<unknown>":
      return t
    if len(shape) == 2:
      return t[:, None, None, :]
    elif len(shape) == 4:
      fmt = property_inference.infer_data_format(t)
      if fmt == "NCHW":
        return tf.transpose(t, [0, 2, 3, 1])
    return t
  return T2


def _T_handle_batch(T, batch=None):
  def T2(name):
    t = T(name)
    if isinstance(batch, int):
      return t[batch:batch+1]
    else:
      return t
  return T2
