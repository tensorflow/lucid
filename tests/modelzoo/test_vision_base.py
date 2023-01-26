import pytest
import tensorflow as tf

from lucid.modelzoo.vision_base import Model
from lucid.modelzoo.vision_models import AlexNet, InceptionV1, InceptionV3_slim, ResnetV1_50_slim


def test_suggest_save_args_happy_path(capsys, minimodel):
  path = "./tests/fixtures/minigraph.pb"

  with tf.Graph().as_default() as graph, tf.compat.v1.Session() as sess:
    _ = minimodel()
    sess.run(tf.compat.v1.global_variables_initializer())

    # ask for suggested arguments
    inferred = Model.suggest_save_args()
    # they should be both printed...
    captured = capsys.readouterr().out  # captures stdout
    names = ["input_name", "image_shape", "output_names"]
    assert all(name in captured for name in names)
    #...and returned

    # check that these inferred values work
    inferred.update(image_value_range=(0,1))
    Model.save(path, **inferred)
    loaded_model = Model.load(path)
    assert "0.100" in repr(loaded_model.graph_def)


def test_suggest_save_args_int_input(capsys, minimodel):
  with tf.Graph().as_default() as graph, tf.compat.v1.Session() as sess:
    image_t = tf.compat.v1.placeholder(tf.uint8, shape=(32, 32, 3), name="input")
    input_t = tf.math.divide(image_t, tf.constant(255, dtype=tf.uint8), name="divide")
    _ = minimodel(input_t)
    sess.run(tf.compat.v1.global_variables_initializer())

    # ask for suggested arguments
    inferred = Model.suggest_save_args()
    captured = capsys.readouterr().out  # captures stdout
    assert "DT_UINT8" in captured
    assert inferred["input_name"] == "divide"


@pytest.mark.parametrize("model_class", [AlexNet, InceptionV1, InceptionV3_slim, ResnetV1_50_slim])
def test_suggest_save_args_existing_graphs(capsys, model_class):
  graph_def = model_class().graph_def

  if model_class == InceptionV1:  # has flexible input shape, can't be inferred
    with pytest.warns(UserWarning):
      inferred = Model.suggest_save_args(graph_def)
  else:
    inferred = Model.suggest_save_args(graph_def)

  assert model_class.input_name == inferred["input_name"]

  if model_class != InceptionV1:
    assert model_class.image_shape == inferred["image_shape"]

  layer_names = [layer.name for layer in model_class.layers]
  for output_name in list(inferred["output_names"]):
    assert output_name in layer_names
