import pytest
import tensorflow as tf

from lucid.modelzoo.vision_base import Model
from lucid.modelzoo.vision_models import AlexNet


shape = (16,16,3)

def test_Model_save():
  with tf.Session().as_default() as sess:
    x = tf.placeholder(tf.float32, shape=shape, name="input")
    w = tf.Variable(0.1, name="variable")
    logits = tf.reduce_mean(w*x, axis=(0,1))
    y = tf.nn.softmax(logits, name="output")
    sess.run(tf.global_variables_initializer())
    path = "./tests/fixtures/minigraph.pb"
    Model.save(path, "input", ["output"], shape, [0,1])

def test_Model_load():
  path = "./tests/fixtures/minigraph.pb"
  model = Model.load(path)
  assert all(str(shape[i]) in repr(model.graph_def) for i in range(len(shape)))
