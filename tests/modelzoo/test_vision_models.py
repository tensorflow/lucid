from __future__ import absolute_import, division, print_function

import pytest

import tensorflow as tf
from lucid.modelzoo.vision_models import InceptionV1

important_layer_names = [
  "mixed3a", "mixed3b", "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e",
  "mixed5a", "mixed5b"
]

def test_InceptionV1_model_download():
  model = InceptionV1()
  model.load_graphdef()
  assert model.graph_def is not None

def test_InceptionV1_graph_import():
  model = InceptionV1()
  model.load_graphdef()
  model.import_graph()
  nodes = tf.get_default_graph().as_graph_def().node
  node_names = set(node.name for node in nodes)
  for layer_name in important_layer_names:
    assert "import/"+layer_name+"_pre_relu" in node_names

def test_InceptionV1_labels():
  model = InceptionV1()
  assert model.labels is not None
  assert model.labels[0] == "dummy"
