from __future__ import absolute_import, division, print_function

import pytest
import inspect
import tensorflow as tf

from lucid.modelzoo.nets_factory import models_map, get_model
from lucid.modelzoo import caffe_models, other_models, slim_models, vision_models
from lucid.modelzoo.vision_base import Layer


clean_modules = [
    caffe_models,
    other_models,
    slim_models,
    vision_models
]

forbidden_names = [
    "obj",
    "name",
    "_obj",
    "_name",
    "absolute_import",
    "division",
    "print_function",
    "IMAGENET_MEAN",
    "IMAGENET_MEAN_BGR",
    "_layers_from_list_of_dicts"
]


@pytest.mark.parametrize("module", clean_modules, ids=lambda m: m.__name__.split('.')[-1])
def test_clean_namespace(module):
    names = dir(module)
    for forbidden in forbidden_names:
      assert forbidden not in names


def test_consistent_namespaces():
    model_names = set(models_map.keys())
    exported_model_names = set(dir(vision_models))
    diffs = model_names.symmetric_difference(exported_model_names)
    for difference in diffs:
        assert difference in ('Model', 'Layer') or difference.startswith("__")


@pytest.mark.slow
@pytest.mark.parametrize("name,model_class", models_map.items())
def test_model_properties(name, model_class):
    assert hasattr(model_class, "model_path")
    assert model_class.model_path.endswith(".pb")
    assert hasattr(model_class, "labels_path")
    assert model_class.labels_path.endswith(".txt")
    assert hasattr(model_class, "dataset")
    assert hasattr(model_class, "image_shape")
    assert len(model_class.image_shape) == 3
    assert hasattr(model_class, "image_value_range")
    assert hasattr(model_class, "input_name")
    assert hasattr(model_class, "layers")
    assert len(model_class.layers) > 0
    last_layer = model_class.layers[-1]
    assert 'dense' in last_layer.tags
    assert type(last_layer) == Layer
    assert last_layer.model_class == model_class
    model_instance = model_class()
    assert model_instance.name == model_class.__name__
    assert last_layer.model_name == model_instance.name

@pytest.mark.slow
@pytest.mark.parametrize("model_class", models_map.values())
def test_model_layers_shapes(model_class):
    name = model_class.__name__
    scope = "TestLucidModelzoo"
    model = model_class()
    with tf.Graph().as_default() as graph:
        model.import_graph(scope=scope)
        for layer in model.layers:
            name, declared_size = layer.name, layer.depth
            imported_name = "{}/{}:0".format(scope, name)
            tensor = graph.get_tensor_by_name(imported_name)
            actual_size = tensor.shape[-1]
            assert int(actual_size) == int(declared_size)
