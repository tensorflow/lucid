from lucid.modelzoo.vision_base import Model, _layers_from_list_of_dicts


class CustomModel(Model):
    """Example of custom Lucid Model class. This example is based on Mobilenet
    from Keras Applications
    """

    model_path = "lucid_protobuf_file.pb"
    dataset = "ImageNet"
    image_shape = [224, 224, 3]
    image_value_range = (-1, 1)
    input_name = "input"
    # Labels as a index-class name dictionnary :
    # Of course if you really use a daset with 1000 classes you 
    # should consider loading them from a file.
    _labels = {
        0: 'tench, Tinca tinca',
        1: 'goldfish, Carassius auratus',
        2: 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
        3: 'tiger shark, Galeocerdo cuvieri',
        4: 'hammerhead, hammerhead shark',
        5: 'electric ray, crampfish, numbfish, torpedo',
        6: 'stingray',
        7: 'cock',
        8: 'hen',
        9: 'ostrich, Struthio camelus',
        # ...
        999: 'toilet tissue, toilet paper, bathroom tissue'}
    }

    @property
    def labels(self):
        return self._labels

    def label_index(self, label):
        return list(self._labels.values()).index(label)

CustomModel.layers = _layers_from_list_of_dicts(
    CustomModel(),
    [
        {"name": "conv1_relu/Relu6", "depth": 32, "tags": ["conv"]},
        {"name": "conv_pw_1_relu/Relu6", "depth": 64, "tags": ["conv"]},
        {"name": "conv_pw_2_relu/Relu6", "depth": 128, "tags": ["conv"]},
        {"name": "conv_pw_3_relu/Relu6", "depth": 128, "tags": ["conv"]},
        {"name": "conv_pw_4_relu/Relu6", "depth": 256, "tags": ["conv"]},
        {"name": "conv_pw_5_relu/Relu6", "depth": 256, "tags": ["conv"]},
        {"name": "conv_pw_6_relu/Relu6", "depth": 512, "tags": ["conv"]},
        {"name": "conv_pw_7_relu/Relu6", "depth": 512, "tags": ["conv"]},
        {"name": "conv_pw_8_relu/Relu6", "depth": 512, "tags": ["conv"]},
        {"name": "conv_pw_9_relu/Relu6", "depth": 512, "tags": ["conv"]},
        {"name": "conv_pw_10_relu/Relu6", "depth": 512, "tags": ["conv"]},
        {"name": "conv_pw_11_relu/Relu6", "depth": 512, "tags": ["conv"]},
        {"name": "conv_pw_12_relu/Relu6", "depth": 1024, "tags": ["conv"]},
        {"name": "conv_pw_13_relu/Relu6", "depth": 1024, "tags": ["conv"]},
        {"name": "dense/BiasAdd", "depth": 256, "tags": ["dense"]},
        {"name": "dense_1/BiasAdd", "depth": 256, "tags": ["dense"]},
        {"name": "dense_2/BiasAdd", "depth": 1000, "tags": ["dense"]},
        {"name": "softmax/Softmax", "depth": 1000, "tags": ["dense"]},
    ],
)

output_shapes = {
    "conv1_relu/Relu6": (112, 112, 32),
    "conv_pw_1_relu/Relu6": (112, 112, 64),
    "conv_pw_2_relu/Relu6": (56, 56, 128),
    "conv_pw_3_relu/Relu6": (56, 56, 128),
    "conv_pw_4_relu/Relu6": (28, 28, 256),
    "conv_pw_5_relu/Relu6": (28, 28, 256),
    "conv_pw_6_relu/Relu6": (14, 14, 512),
    "conv_pw_7_relu/Relu6": (14, 14, 512),
    "conv_pw_8_relu/Relu6": (14, 14, 512),
    "conv_pw_9_relu/Relu6": (14, 14, 512),
    "conv_pw_10_relu/Relu6": (14, 14, 512),
    "conv_pw_11_relu/Relu6": (14, 14, 512),
    "conv_pw_12_relu/Relu6": (7, 7, 1024),
    "conv_pw_13_relu/Relu6": (7, 7, 1024),
    "dense/BiasAdd": (256,),
    "dense_1/BiasAdd": (256,),
    "dense_2/BiasAdd": (1000,),
    "softmax/Softmax": (1000,),
}

for layer in CustomModel.layers:
    layer.shape = output_shapes[layer.name]
