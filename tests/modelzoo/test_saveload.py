import pytest
import tensorflow as tf

from lucid.modelzoo.vision_base import Model
from lucid.modelzoo.vision_models import AlexNet


shape = (16,16,3)

def test_Model_save(minimodel):
  with tf.compat.v1.Session().as_default() as sess:
    _ = minimodel()
    sess.run(tf.compat.v1.global_variables_initializer())
    path = "./tests/fixtures/minigraph.pb"
    Model.save(path, "input", ["output"], shape, [0,1])

def test_Model_load():
  path = "./tests/fixtures/minigraph.pb"
  model = Model.load(path)
  assert all(str(shape[i]) in repr(model.graph_def) for i in range(len(shape)))
