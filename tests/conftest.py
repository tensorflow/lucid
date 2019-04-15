import pytest
import tensorflow as tf


@pytest.fixture
def minimodel():
  def inner(input=None, shape=(16,16,3)):
    """Constructs a tiny graph containing one each of a typical input
    (tf.placegholder), variable and typical output (softmax) nodes."""
    if input is None:
      input = tf.placeholder(tf.float32, shape=shape, name="input")
    w = tf.Variable(0.1, name="variable")
    logits = tf.reduce_mean(w*input, name="output", axis=(0,1))
    return tf.nn.softmax(logits)
  return inner

# Add support for a slow tests marker:


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    else:
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
