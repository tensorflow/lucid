# Copyright 2018 The Lucid Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Objective functions for visualizing neural networks.

We represent objectives with a class `Objective` enclsoing functions of the
form:

  (T) => TensorFlow Scalar

Where `T` is a function that allows one to access the activations of different
layers of the network. For example `T("mixed4a")` gives the activations for
the layer mixed4a.

This allows objectives to be declared outside the rendering function, but then
actually constructed within its graph/session.
"""

from __future__ import absolute_import, division, print_function

from decorator import decorator
import numpy as np
import tensorflow as tf

# We use T as a variable name to access all kinds of tensors
# pylint: disable=invalid-name


class Objective(object):
  """"A wrapper to make objective functions easy to combine.

  For example, suppose you want to optimize 20% for mixed4a:20 and 80% for
  mixed4a:21. Then you could use:

    objetive = 0.2 * channel("mixed4a", 20) + 0.8 * channel("mixed4a", 0.8)

  Under the hood, we think of objectives as functions of the form:

    T => tensorflow scalar for loss

  where T is a function allowing you to index layers in the network -- that is,
  if there's a layer "mixed4a" then T("mixed4a") would give you its
  activations).

  This allows objectives to be declared outside the rendering function, but then
  actually constructed within its graph/session.
  """

  def __init__(self, objective_func, name="", description=""):
    self.objective_func = objective_func
    self.name = name
    self.description = description

  def __add__(self, other):
    if isinstance(other, (int, float)):
      objective_func = lambda T: other + self(T)
      name = self.name
      description = self.description
    else:
      objective_func = lambda T: self(T) + other(T)
      name = ", ".join([self.name, other.name])
      description = "Sum(" + " +\n".join([self.description, other.description]) + ")"
    return Objective(objective_func, name=name, description=description)

  def __neg__(self):
    return -1 * self

  def __sub__(self, other):
    return self + (-1 * other)

  @staticmethod
  def sum(objs):
    objective_func = lambda T: sum([obj(T) for obj in objs])
    descriptions = [obj.description for obj in objs]
    description = "Sum(" + " +\n".join(descriptions) + ")"
    names = [obj.name for obj in objs]
    name = ", ".join(names)
    return Objective(objective_func, name=name, description=description)

  def __mul__(self, other):
    if isinstance(other, (int, float)):
      objective_func = lambda T: other * self(T)
    else:
      objective_func = lambda T: self(T) * other(T)
    return Objective(objective_func, name=self.name, description=self.description)

  def __rmul__(self, other):
    return self.__mul__(other)

  def __radd__(self, other):
    return self.__add__(other)

  def __call__(self, T):
    return self.objective_func(T)


def _make_arg_str(arg):
  arg = str(arg)
  too_big = len(arg) > 15 or "\n" in arg
  return "..." if too_big else arg


@decorator
def wrap_objective(f, *args, **kwds):
  """Decorator for creating Objective factories.

  Changes f from the closure: (args) => () => TF Tensor
  into an Obejective factory: (args) => Objective

  while perserving function name, arg info, docs... for interactive python.
  """
  objective_func = f(*args, **kwds)
  objective_name = f.__name__
  args_str = " [" + ", ".join([_make_arg_str(arg) for arg in args]) + "]"
  description = objective_name.title() + args_str
  return Objective(objective_func, objective_name, description)


@wrap_objective
def neuron(layer_name, channel_n, x=None, y=None, batch=None):
  """Visualize a single neuron of a single channel."""
  def inner(T):
    layer = T(layer_name)
    shape = tf.shape(layer)
    if x is None:
      x_ = shape[1] // 2
    if y is None:
      y_ = shape[2] // 2
    if batch is None:
      return layer[:, x_, y_, channel_n]
    else:
      return layer[batch, x_, y_, channel_n]
  return inner


@wrap_objective
def channel(layer, n_channel, batch=None):
  """Visualize a single channel"""
  if batch is None:
    return lambda T: tf.reduce_mean(T(layer)[..., n_channel])
  else:
    return lambda T: tf.reduce_mean(T(layer)[batch, ..., n_channel])


@wrap_objective
def direction(layer, vec, batch=None):
  """Visualize a direction"""
  if batch is None:
    vec = vec[None, None, None]
    return lambda T: tf.reduce_mean(T(layer) * vec)
  else:
    vec = vec[None, None]
    return lambda T: tf.reduce_mean(T(layer)[batch] * vec)


@wrap_objective
def direction_neuron(layer_name, vec, batch=None, x=None, y=None):
  """Visualize a single (x, y) position along the given direction"""
  def inner(T):
    layer = T(layer_name)
    shape = tf.shape(layer)
    x_ = shape[1] / 2 if x is None else x
    y_ = shape[2] / 2 if y is None else y
    if batch is None:
      return tf.reduce_mean(layer[:, x_, y_] * vec[None])
    else:
      return tf.reduce_mean(layer[batch, x_, y_] * vec)
  return inner

@wrap_objective
def direction_cossim(layer, vec, batch=None):
  """Visualize a direction (cossine similarity)"""
  def inner(T):
    act_mags = tf.sqrt(tf.reduce_sum(T(layer)**2, -1, keep_dims=True))
    vec_mag = tf.sqrt(tf.reduce_sum(vec**2))
    mags = act_mags * vec_mag
    if batch is None:
      return tf.reduce_mean(T(layer) * vec.reshape([1, 1, 1, -1]) / mags)
    else:
      return tf.reduce_mean(T(layer)[batch] * vec.reshape([1, 1, -1]) / mags)
  return inner


@wrap_objective
def deepdream(layer):
  """Maximize 'interestingness' at some layer.

  See Mordvintsev et al., 2015.
  """
  return lambda T: tf.reduce_mean(T(layer)**2)


@wrap_objective
def total_variation(layer="input"):
  """Total variation of image (or activations at some layer).

  This operation is most often used as a penalty to reduce noise.
  See Simonyan, et al., 2014.
  """
  return lambda T: tf.image.total_variation(T(layer))


@wrap_objective
def L1(layer="input", constant=0, batch=None):
  """L1 norm of layer. Generally used as penalty."""
  if batch is None:
    return lambda T: tf.reduce_sum(tf.abs(T(layer) - constant))
  else:
    return lambda T: tf.reduce_sum(tf.abs(T(layer)[batch] - constant))


@wrap_objective
def L2(layer="input", constant=0, epsilon=1e-6, batch=None):
  """L2 norm of layer. Generally used as penalty."""
  if batch is None:
    return lambda T: tf.sqrt(epsilon + tf.reduce_sum((T(layer) - constant) ** 2))
  else:
    return lambda T: tf.sqrt(epsilon + tf.reduce_sum((T(layer)[batch] - constant) ** 2))


def _tf_blur(x, w=3):
  depth = x.shape[-1]
  k = np.zeros([w, w, depth, depth])
  for ch in range(depth):
    k_ch = k[:, :, ch, ch]
    k_ch[ :,    :  ] = 0.5
    k_ch[1:-1, 1:-1] = 1.0

  conv_k = lambda t: tf.nn.conv2d(t, k, [1, 1, 1, 1], "SAME")
  return conv_k(x) / conv_k(tf.ones_like(x))


@wrap_objective
def blur_input_each_step():
  """Minimizing this objective is equivelant to blurring input each step.

  Optimizing (-k)*blur_input_each_step() is equivelant to:

    input <- (1-k)*input + k*blur(input)

  An operation that was used in early feature visualization work.
  See Nguyen, et al., 2015.
  """
  def inner(T):
    t_input = T("input")
    t_input_blurred = tf.stop_gradient(_tf_blur(t_input))
    return 0.5*tf.reduce_sum((t_input - t_input_blurred)**2)
  return inner

@wrap_objective
def blur_alpha_each_step():
  """Minimizing this objective is equivelant to blurring input each step.

  Optimizing (-k)*blur_input_each_step() is equivelant to:

    input <- (1-k)*input + k*blur(input)

  An operation that was used in early feature visualization work.
  See Nguyen, et al., 2015.
  """
  def inner(T):
    t_input = T("input")[..., 3:4]
    t_input_blurred = tf.stop_gradient(_tf_blur(t_input))
    return 0.5*tf.reduce_sum((t_input - t_input_blurred)**2)
  return inner


@wrap_objective
def channel_interpolate(layer1, n_channel1, layer2, n_channel2):
  """Interpolate between layer1, n_channel1 and layer2, n_channel2.

  Optimize for a convex combination of layer1, n_channel1 and
  layer2, n_channel2, transitioning across the batch.

  Args:
    layer1: layer to optimize 100% at batch=0.
    n_channel1: neuron index to optimize 100% at batch=0.
    layer2: layer to optimize 100% at batch=N.
    n_channel2: neuron index to optimize 100% at batch=N.

  Returns:
    Objective
  """
  def inner(T):
    batch_n = T(layer1).get_shape().as_list()[0]
    arr1 = T(layer1)[..., n_channel1]
    arr2 = T(layer2)[..., n_channel2]
    weights = (np.arange(batch_n)/float(batch_n-1))
    S = 0
    for n in range(batch_n):
      S += (1-weights[n]) * tf.reduce_mean(arr1[n])
      S += weights[n] * tf.reduce_mean(arr2[n])
    return S
  return inner


@wrap_objective
def penalize_boundary_complexity(shp, w=20, mask=None, C=0.5):
  """Encourage the boundaries of an image to have less variation and of color C.

  Args:
    shp: shape of T("input") because this may not be known.
    w: width of boundary to penalize. Ignored if mask is set.
    mask: mask describing what area should be penalized.

  Returns:
    Objective.
  """
  def inner(T):
    arr = T("input")

    # print shp
    if mask is None:
      mask_ = np.ones(shp)
      mask_[:, w:-w, w:-w] = 0
    else:
      mask_ = mask

    blur = _tf_blur(arr, w=5)
    diffs = (blur-arr)**2
    diffs += 0.8*(arr-C)**2

    return -tf.reduce_sum(diffs*mask_)
  return inner


@wrap_objective
def alignment(layer, decay_ratio=2):
  """Encourage neighboring images to be similar.

  When visualizing the interpolation between two objectives, it's often
  desireable to encourage analagous boejcts to be drawn in the same position,
  to make them more comparable.

  This term penalizes L2 distance between neighboring images, as evaluated at
  layer.

  In general, we find this most effective if used with a paramaterization that
  shares across the batch. (In fact, that works quite well by iteself, so this
  function may just be obselete.)

  Args:
    layer: layer to penalize at.
    decay_ratio: how much to decay penalty as images move apart in batch.

  Returns:
    Objective.
  """
  def inner(T):
    batch_n = T(layer).get_shape().as_list()[0]
    arr = T(layer)
    accum = 0
    for d in [1, 2, 3, 4]:
      for i in range(batch_n - d):
        a, b = i, i+d
        arr1, arr2 = arr[a], arr[b]
        accum += tf.reduce_mean((arr1-arr2)**2) / decay_ratio**float(d)
    return -accum
  return inner

@wrap_objective
def diversity(layer):
  """Encourage diversity between each batch element.

  A neural net feature often responds to multiple things, but naive feature
  visualization often only shows us one. If you optimize a batch of images,
  this objective will encourage them all to be different.

  In particular, it caculuates the correlation matrix of activations at layer
  for each image, and then penalizes cossine similarity between them. This is
  very similar to ideas in style transfer, except we're *penalizing* style
  similarity instead of encouraging it.

  Args:
    layer: layer to evaluate activation correlations on.

  Returns:
    Objective.
  """
  def inner(T):
    layer_t = T(layer)
    batch_n, _, _, channels = layer_t.get_shape().as_list()

    flattened = tf.reshape(layer_t, [batch_n, -1, channels])
    grams = tf.matmul(flattened, flattened, transpose_a=True)
    grams = tf.nn.l2_normalize(grams, dim=[1,2], epsilon=1e-10)

    return sum([ sum([ tf.reduce_sum(grams[i]*grams[j])
                      for j in range(batch_n) if j != i])
                for i in range(batch_n)]) / batch_n
  return inner


@wrap_objective
def input_diff(orig_img):
  """Average L2 difference between optimized image and orig_img.

  This objective is usually mutliplied by a negative number and used as a
  penalty in making advarsarial counterexamples.
  """
  def inner(T):
    diff = T("input") - orig_img
    return tf.sqrt(tf.reduce_mean(diff**2))
  return inner


@wrap_objective
def class_logit(layer, label):
  """Like channel, but for softmax layers.

  Args:
    layer: A layer name string.
    label: Either a string (refering to a label in model.labels) or an int
      label position.

  Returns:
    Objective maximizing a logit.
  """
  def inner(T):
    if isinstance(label, int):
      class_n = label
    else:
      class_n = T("labels").index(label)
    logits = T(layer)
    logit = tf.reduce_sum(logits[:, class_n])
    return logit
  return inner


def as_objective(obj):
  """Convert obj into Objective class.

  Strings of the form "layer:n" become the Objective channel(layer, n).
  Objectives are returned unchanged.

  Args:
    obj: string or Objective.

  Returns:
    Objective
  """
  if isinstance(obj, Objective):
    return obj
  elif callable(obj):
    return obj
  elif isinstance(obj, str):
    layer, n = obj.split(":")
    layer, n = layer.strip(), int(n)
    return channel(layer, n)
