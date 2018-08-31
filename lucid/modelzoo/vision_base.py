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

from __future__ import absolute_import, division, print_function
from future.utils import with_metaclass
from os import path

import tensorflow as tf
import numpy as np

from lucid.modelzoo.util import load_text_labels, load_graphdef, forget_xy
from lucid.modelzoo.aligned_activations import get_aligned_activations as _get_aligned_activations
from lucid.misc.io import load
import lucid.misc.io.showing as showing

IMAGENET_MEAN = np.array([123.68, 116.779, 103.939])
IMAGENET_MEAN_BGR = np.flip(IMAGENET_MEAN, 0)


import warnings


class Layer(object):
  """Layer provides information on a model's layers."""

  width = None  # reserved for future use
  height = None  # reserved for future use
  shape = None  # reserved for future use

  def __init__(self, model_class, name, depth, tags):
    self._activations = None
    self.model_class = model_class
    self.name = name
    self.depth = depth
    self.tags = set(tags)

  def __getitem__(self, name):
    if name == 'type':
      warnings.warn("Property 'type' is deprecated on model layers. Please check if 'tags' contains the type you are looking for in the future! We're simply a tag for now.", DeprecationWarning)
      return list(self.tags)[0]
    if name not in self.__dict__:
      error_message = "'Layer' object has no attribute '{}'".format(name)
      raise AttributeError(error_message)
    return self.__dict__[name]

  @property
  def size(self):
    warnings.warn("Property 'size' is deprecated on model layers because it may be confused with the spatial 'size' of a layer. Please use 'depth' in the future!", DeprecationWarning)
    return self.depth

  @property
  def activations(self):
    """Loads sampled activations, which requires network access."""
    if self._activations is None:
      self._activations = _get_aligned_activations(self)
    return self._activations

  def __repr__(self):
    return "Layer (belonging to {s.model_class.name}) <{s.name}: {s.depth}> ([{s.tags}])".format(s=self)


def _layers_from_list_of_dicts(model_class, list_of_dicts):
  layers = []
  for layer_info in list_of_dicts:
    name, depth, tags = layer_info['name'], layer_info['depth'], layer_info['tags']
    layer = Layer(model_class, name, depth, tags)
    layers.append(layer)
  return layers


class ModelPropertiesMetaClass(type):

  @property
  def name(cls):
      return cls.__name__


class Model(with_metaclass(ModelPropertiesMetaClass, object)):
  """Model allows importing pre-trained models."""

  model_path = None
  labels_path = None
  labels = None
  image_value_range = (-1, 1)
  image_shape = [None, None, 3]
  layers = []

  def __init__(self):
    self.graph_def = None
    if self.labels_path is not None:
      self.labels = load_text_labels(self.labels_path)

  @property
  def name(self):
    return self.__class__.name

  def load_graphdef(self):
    self.graph_def = load_graphdef(self.model_path)

  def post_import(self, scope):
    pass

  def create_input(self, t_input=None, forget_xy_shape=True):
    """Create input tensor."""
    if t_input is None:
      t_input = tf.placeholder(tf.float32, self.image_shape)
    t_prep_input = t_input
    if len(t_prep_input.shape) == 3:
      t_prep_input = tf.expand_dims(t_prep_input, 0)
    if forget_xy_shape:
      t_prep_input = forget_xy(t_prep_input)
    if hasattr(self, "is_BGR") and self.is_BGR == True:
      t_prep_input = tf.reverse(t_prep_input, [-1])
    lo, hi = self.image_value_range
    t_prep_input = lo + t_prep_input * (hi-lo)
    return t_input, t_prep_input

  def import_graph(self, t_input=None, scope='import', forget_xy_shape=True):
    """Import model GraphDef into the current graph."""
    if self.graph_def is None:
      raise Exception("Model.import_graph(): Must load graph def before importing it.")
    graph = tf.get_default_graph()
    assert graph.unique_name(scope, False) == scope, (
        'Scope "%s" already exists. Provide explicit scope names when '
        'importing multiple instances of the model.') % scope
    t_input, t_prep_input = self.create_input(t_input, forget_xy_shape)
    tf.import_graph_def(
        self.graph_def, {self.input_name: t_prep_input}, name=scope)
    self.post_import(scope)

  def show_graph(self):
    if self.graph_def is None:
      raise Exception("Model.show_graph(): Must load graph def before showing it.")
    showing.graph(self.graph_def)


class SerializedModel(Model):
  """Allows importing various types of serialized models from a directory.

  (Currently only supports frozen graph models and relies on manifest.json file.
  In the future we may want to support automatically detecting the type and
  support loading more ways of saving models: tf.SavedModel, metagraphs, etc.)
  """

  @classmethod
  def from_directory(cls, model_path):
    manifest_path = path.join(model_path, 'manifest.json')
    try:
      manifest = load(manifest_path)
    except Exception as e:
      raise ValueError("Could not find manifest.json file in dir {}. Error: {}".format(model_path, e))
    if manifest.get('type', 'frozen') == 'frozen':
      return FrozenGraphModel(model_path, manifest)
    else: # TODO: add tf.SavedModel support, etc
      raise NotImplementedError("SerializedModel Manifest type '{}' has not been implemented!".format(manifest.get('type')))


class FrozenGraphModel(SerializedModel):

  def __init__(self, model_directory, manifest):
    model_path = manifest.get('model_path', 'graph.pb')
    if model_path.startswith("./"): # TODO: can we be less specific here?
      self.model_path = path.join(model_directory, model_path)
    else:
      self.model_path = model_path
    self.labels_path = manifest.get('labels_path', None)
    self.image_value_range = manifest.get('image_value_range', None)
    self.image_shape = manifest.get('image_shape', None)
    self.input_name = manifest.get('input_name', 'input:0')
    super().__init__()
