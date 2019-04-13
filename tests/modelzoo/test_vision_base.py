import tensorflow as tf

from lucid.modelzoo.vision_base import Model

shape = [16,16,3]

def test_suggest_save_args(capsys):
  path = "./tests/fixtures/minigraph.pb"

  with tf.Graph().as_default() as graph, tf.Session() as sess:
    x = tf.placeholder(tf.float32, shape=shape, name="input")
    w = tf.Variable(0.1, name="variable")
    logits = tf.reduce_mean(w*x, name="output", axis=(0,1))
    y = tf.nn.softmax(logits)
    sess.run(tf.global_variables_initializer())

    # ask for suggested arguments
    inferred = Model.suggest_save_args()

    # they should be both printed...
    captured = capsys.readouterr().out  # captures stdout
    names = ["input_name", "image_shape", "output_names"]
    assert all(name in captured for name in names)
    #...and returned
    assert all(inferred.values())

    # check that these inferred values work
    Model.save(path, image_value_range=(0,1), **inferred)
    loaded_model = Model.load(path)
    assert str(shape) in repr(loaded_model.graph_def)

