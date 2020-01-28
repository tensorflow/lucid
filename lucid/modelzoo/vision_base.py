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
from os import path
import warnings
import logging
from itertools import chain

import tensorflow as tf
import numpy as np

from lucid.modelzoo import util as model_util
from lucid.modelzoo.aligned_activations import get_aligned_activations as _get_aligned_activations
from lucid.modelzoo.get_activations import get_activations as _get_activations
from lucid.misc.io import load, save
import lucid.misc.io.showing as showing


IMAGENET_MEAN = np.array([123.68, 116.779, 103.939])
IMAGENET_MEAN_BGR = np.flip(IMAGENET_MEAN, 0)


log = logging.getLogger(__name__)


class Layer(object):
  """Layer provides information on a model's layers."""

  width = None  # reserved for future use
  height = None  # reserved for future use
  shape = None  # reserved for future use

  def __init__(self, model_instance: 'Model', name, depth, tags):
    self._activations = None
    self.model_class = model_instance.__class__
    self.model_name = model_instance.name
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
    return f"Layer (belonging to {self.model_name}) <{self.name}: {self.depth}> ([{self.tags}])"

  def to_json(self):
    return self.name  # TODO


def _layers_from_list_of_dicts(model_instance: 'Model', list_of_dicts):
  layers = []
  for layer_info in list_of_dicts:
    name, depth, tags = layer_info['name'], layer_info['depth'], layer_info['tags']
    layer = Layer(model_instance, name, depth, tags)
    layers.append(layer)
  return tuple(layers)


class Model():
  """Model allows using pre-trained models."""

  model_path = None
  labels_path = None
  image_value_range = (-1, 1)
  image_shape = (None, None, 3)
  layers = ()
  model_name = None

  _labels = None
  _synset_ids = None
  _synsets = None
  _graph_def = None

  # Avoid pickling the in-memory graph_def.
  _blacklist = ['_graph_def']
  def __getstate__(self):
      return {k: v for k, v in self.__dict__.items() if k not in self._blacklist}

  def __setstate__(self, state):
      self.__dict__.update(state)

  def __eq__(self, other):
    if isinstance(other, Model):
        return self.model_path == other.model_path
    return False

  def __hash__(self):
    return hash(self.model_path)

  @property
  def labels(self):
    if not hasattr(self, 'labels_path') or self.labels_path is None:
      raise RuntimeError("This model does not have a labels_path specified!")
    if not self._labels:
      self._labels = load(self.labels_path, split=True)
    return self._labels

  @property
  def synset_ids(self):
    if not hasattr(self, 'synsets_path') or self.synsets_path is None:
      raise RuntimeError("This model does not have a synset_path specified!")
    if not self._synset_ids:
      self._synset_ids = load(self.synsets_path, split=True)
    return self._synset_ids

  @property
  def synsets(self):
    from lucid.modelzoo.wordnet import synset_from_id

    if not self._synsets:
      self._synsets = [synset_from_id(s_id) for s_id in self.synset_ids]
    return self._synsets

  @property
  def name(self):
    if self.model_name == None:
      return self.__class__.__name__
    else:
      return self.model_name

  def __str__(self):
    return self.name

  def to_json(self):
    return self.name  # TODO

  @property
  def graph_def(self):
    if not self._graph_def:
      self._graph_def = model_util.load_graphdef(self.model_path)
    return self._graph_def

  def load_graphdef(self):
    warnings.warn(
        "Calling `load_graphdef` is no longer necessary and now a noop. Graphs are loaded lazily when a models graph_def property is accessed.",
        DeprecationWarning,
    )
    pass

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
      t_prep_input = model_util.forget_xy(t_prep_input)
    if hasattr(self, "is_BGR") and self.is_BGR is True:
      t_prep_input = tf.reverse(t_prep_input, [-1])
    lo, hi = self.image_value_range
    t_prep_input = lo + t_prep_input * (hi - lo)
    return t_input, t_prep_input

  def import_graph(self, t_input=None, scope='import', forget_xy_shape=True, input_map=None):
    """Import model GraphDef into the current graph."""
    graph = tf.get_default_graph()
    assert graph.unique_name(scope, False) == scope, (
        'Scope "%s" already exists. Provide explicit scope names when '
        'importing multiple instances of the model.') % scope
    t_input, t_prep_input = self.create_input(t_input, forget_xy_shape)
    final_input_map = {self.input_name: t_prep_input}
    if input_map is not None:
      final_input_map.update(input_map)
    tf.import_graph_def(
        self.graph_def, final_input_map, name=scope)
    self.post_import(scope)

    def T(layer):
      if ":" in layer:
          return graph.get_tensor_by_name("%s/%s" % (scope,layer))
      else:
          return graph.get_tensor_by_name("%s/%s:0" % (scope,layer))

    return T

  def show_graph(self):
    if self.graph_def is None:
      raise Exception("Model.show_graph(): Must load graph def before showing it.")
    showing.graph(self.graph_def)

  def get_layer(self, name):
    # Search by exact match
    for layer in self.layers:
      if layer.name == name:
        return layer
    # if not found by exact match, search fuzzy and warn user:
    for layer in self.layers:
      if name.lower() in layer.name.lower():
        log.warning("Found layer by fuzzy matching, please use '%s' in the future!", layer.name)
        return layer
    key_error_message = "Could not find layer with name '{}'! Existing layer names are: {}"
    layer_names = str([l.name for l in self.layers])
    raise KeyError(key_error_message.format(name, layer_names))

  @staticmethod
  def suggest_save_args(graph_def=None):
    if graph_def is None:
      graph_def = tf.get_default_graph().as_graph_def()
    gdhelper = model_util.GraphDefHelper(graph_def)
    inferred_info = dict.fromkeys(("input_name", "image_shape", "output_names", "image_value_range"))
    node_shape = lambda n: [dim.size for dim in n.attr['shape'].shape.dim]
    potential_input_nodes = gdhelper.by_op["Placeholder"]
    output_nodes = [node.name for node in gdhelper.by_op["Softmax"]]

    if len(potential_input_nodes) == 1:
      input_node = potential_input_nodes[0]
      input_dtype = tf.dtypes.as_dtype(input_node.attr['dtype'].type)
      if input_dtype.is_floating:
        input_name = input_node.name
        print("Inferred: input_name = {} (because it was the only Placeholder in the graph_def)".format(input_name))
        inferred_info["input_name"] = input_name
      else:
        print("Warning: found a single Placeholder, but its dtype is {}. Lucid's parameterizations can only replace float dtypes. We're now scanning to see if you maybe divide this placeholder by 255 to get a float later in the graph...".format(str(input_node.attr['dtype']).strip()))
        neighborhood = gdhelper.neighborhood(input_node, degree=5)
        divs = [n for n in neighborhood if n.op == "RealDiv"]
        consts = [n for n in neighborhood if n.op == "Const"]
        magic_number_present = any(255 in c.attr['value'].tensor.int_val for c in consts)
        if divs and magic_number_present:
          if len(divs) == 1:
            input_name = divs[0].name
            print("Guessed: input_name = {} (because it's the only division by 255 near the only placeholder)".format(input_name))
            inferred_info["input_name"] = input_name
            image_value_range = (0,1)
            print("Guessed: image_value_range = {} (because you're dividing by 255 near the only placeholder)".format(image_value_range))
            inferred_info["image_value_range"] = (0,1)
          else:
            warnings.warn("Could not infer input_name because there were multiple division ops near your the only placeholder. Candidates include: {}".format([n.name for n in divs]))
    else:
      warnings.warn("Could not infer input_name because there were multiple or no Placeholders.")

    if inferred_info["input_name"] is not None:
      input_node = gdhelper.by_name[inferred_info["input_name"]]
      shape = node_shape(input_node)
      if len(shape) in (3,4):
        if len(shape) == 4:
          shape = shape[1:]
        if -1 not in shape:
          print("Inferred: image_shape = {}".format(shape))
          inferred_info["image_shape"] = shape
      if inferred_info["image_shape"] is None:
        warnings.warn("Could not infer image_shape.")

    if output_nodes:
      print("Inferred: output_names = {}  (because those are all the Softmax ops)".format(output_nodes))
      inferred_info["output_names"] = output_nodes
    else:
      warnings.warn("Could not infer output_names.")

    report = []
    report.append("# Please sanity check all inferred values before using this code.")
    report.append("# Incorrect `image_value_range` is the most common cause of feature visualization bugs! Most methods will fail silently with incorrect visualizations!")
    report.append("Model.save(")

    suggestions = {
        "input_name" : 'input',
        "image_shape" : [224, 224, 3],
        "output_names": ['logits'],
        "image_value_range": "[-1, 1], [0, 1], [0, 255], or [-117, 138]"
    }
    for key, value in inferred_info.items():
      if value is not None:
        report.append("    {}={!r},".format(key, value))
      else:
        report.append("    {}=_,                   # TODO (eg. {!r})".format(key, suggestions[key]))
    report.append("  )")

    print("\n".join(report))
    return inferred_info


  @staticmethod
  def save(save_url, input_name, output_names, image_shape, image_value_range):
    if ":" in input_name:
      raise ValueError("input_name appears to be a tensor (name contains ':') but must be an op.")
    if any([":" in name for name in output_names]):
      raise ValueError("output_namess appears to be contain tensor (name contains ':') but must be ops.")

    metadata = {
      "input_name" : input_name,
      "image_shape" : image_shape,
      "image_value_range": image_value_range,
    }

    graph_def = model_util.frozen_default_graph_def([input_name], output_names)
    model_util.infuse_metadata(graph_def, metadata)
    save(graph_def, save_url)

  @staticmethod
  def load(url):
    if url.endswith(".pb"):
      return Model.load_from_graphdef(url)
    elif url.endswith(".json"):
      return Model.load_from_manifest(url)

  @staticmethod
  def load_from_metadata(model_url, metadata):
    class DynamicModel(Model):
      model_path = model_url
      input_name = metadata["input_name"]
      image_shape = metadata["image_shape"]
      image_value_range = metadata["image_value_range"]
    return DynamicModel()

  @staticmethod
  def load_from_graphdef(graphdef_url):
    graph_def = load(graphdef_url)
    metadata = model_util.extract_metadata(graph_def)
    if metadata:
      return Model.load_from_metadata(graphdef_url, metadata)
    else:
      raise ValueError("Model.load_from_graphdef was called on a GraphDef ({}) that does not contain Lucid's metadata node. Model.load only works for models saved via Model.save. For the graphdef you're trying to load, you will need to provide custom metadata; see Model.load_from_metadata()".format(graphdef_url))

  @staticmethod
  def load_from_manifest(manifest_url):
    try:
      manifest = load(manifest_url)
    except Exception as e:
      raise ValueError("Could not find manifest.json file in dir {}. Error: {}".format(manifest_url, e))

    if manifest.get('type', 'frozen') == 'frozen':
      manifest_folder = path.dirname(manifest_url)
      return FrozenGraphModel(manifest_folder, manifest)
    else:
      raise NotImplementedError("SerializedModel Manifest type '{}' has not been implemented!".format(manifest.get('type')))

  def get_activations(self, layer, examples, batch_size=64,
                         dtype=None, ind_shape=None, center_only=False):
    """Collect center activtions of a layer over an n-dimensional array of images.

    Note: this is mostly intended for large synthetic families of images, where
      you can cheaply generate them in Python. For collecting activations over,
      say, ImageNet, there will be better workflows based on various dataset APIs
      in TensorFlow.

    Args:
      layer: layer (in model) for activtions to be collected from.
      examples: A (potentially n-dimensional) array of images. Can be any nested
        iterable object, including a generator, as long as the inner most objects
        are a numpy array with at least 3 dimensions (image X, Y, channels=3).
      batch_size: How many images should be processed at once?
      dtype: determines dtype of returned data (defaults to model activation
        dtype). Can be used to make function memory efficient.
      ind_shape: Shape that the index (non-image) dimensions of examples. Makes
        code much more memory efficient if examples is not a numpy array.

    Memory efficeincy:
      Have examples be a generator rather than an array of images; this allows
      them to be lazily generated and not all stored in memory at once. Also
      use ind_shape so that activations can be stored in an efficient data
      structure. If you still have memory problems, dtype="float16" can probably
      get you another 2x.

    Returns:
      A numpy array of shape [ind1, ind2, ..., layer_channels]
    """

    return _get_activations(self, layer, examples, batch_size=batch_size,
                            dtype=dtype, ind_shape=ind_shape,
                            center_only=center_only)


class SerializedModel(Model):

  @classmethod
  def from_directory(cls, model_path, manifest_path=None):
    warnings.warn("SerializedModel is deprecated. Please use Model.load_from_manifest instead.", DeprecationWarning)
    if manifest_path is None:
      manifest_path = path.join(model_path, 'manifest.json')
    return Model.load_from_manifest(manifest_path)


class FrozenGraphModel(SerializedModel):

  _mandatory_properties = ['model_path', 'image_value_range', 'input_name', 'image_shape']

  def __init__(self, model_directory, manifest):
    self.manifest = manifest

    for mandatory_key in self._mandatory_properties:
      # TODO: consider if we can tell you the path of the faulty manifest here
      assert mandatory_key in manifest.keys(), "Mandatory property '{}' was not defined in json manifest.".format(mandatory_key)
    for key, value in manifest.items():
      setattr(self, key, value)

    model_path = manifest.get('model_path', 'graph.pb')
    if model_path.startswith("./"): # TODO: can we be less specific here?
      self.model_path = path.join(model_directory, model_path[2:])
    else:
      self.model_path = model_path

    layers_or_layer_names = manifest.get('layers')
    if len(layers_or_layer_names) > 0:
      if isinstance(layers_or_layer_names[0], str):
        self.layer_names = layers_or_layer_names
      elif isinstance(layers_or_layer_names[0], dict):
        self.layers = _layers_from_list_of_dicts(self, layers_or_layer_names)
        self.layer_names = [layer.name for layer in self.layers]

    super().__init__()
