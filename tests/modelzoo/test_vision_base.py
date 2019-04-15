import tensorflow as tf

from lucid.modelzoo.vision_base import Model


def test_suggest_save_args(capsys, minimodel):
  path = "./tests/fixtures/minigraph.pb"

  with tf.Graph().as_default() as graph, tf.Session() as sess:
    _ = minimodel()
    sess.run(tf.global_variables_initializer())

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


