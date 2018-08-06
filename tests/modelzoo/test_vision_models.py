from __future__ import absolute_import, division, print_function

import pytest
import tensorflow as tf

from lucid.modelzoo.nets_factory import models_map, get_model


@pytest.mark.slow
@pytest.mark.parametrize("name,model_class", models_map.items())
def test_model_layers_shapes(name, model_class):
    scope = "TestLucidModelzoo"
    model = model_class()
    model.load_graphdef()
    with tf.Graph().as_default() as graph:
        model.import_graph(scope=scope)
        for layer in model.layers:
            name, declared_size = (layer[key] for key in ("name", "size"))
            imported_name = "{}/{}:0".format(scope, name)
            tensor = graph.get_tensor_by_name(imported_name)
            actual_size = tensor.shape[-1]
            assert int(actual_size) == int(declared_size)
