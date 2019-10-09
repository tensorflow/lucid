# Copyright 2018 The Lucid Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import, division, print_function
from lucid.modelzoo.vision_base import Model, _layers_from_list_of_dicts, IMAGENET_MEAN


class InceptionV1_slim(Model):
  """InceptionV1 as implemented by the TensorFlow slim framework.

  InceptionV1 was introduced by Szegedy, et al (2014):
  https://arxiv.org/pdf/1409.4842v1.pdf
  This function provides the pre-trained reimplementation from TF slim:
  https://github.com/tensorflow/models/tree/master/research/slim
  corresponding to the name "inception_v1".
  """
  layers = None
  model_path  = 'gs://modelzoo/vision/slim_models/InceptionV1.pb'
  labels_path = 'gs://modelzoo/labels/ImageNet_standard_with_dummy.txt'
  synsets_path = 'gs://modelzoo/labels/ImageNet_standard_with_dummy_synsets.txt'
  dataset = 'ImageNet'
  image_shape = [224, 224, 3]
  # inpute range taken from:
  # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py#L280
  image_value_range = (-1, 1)
  input_name = 'input'

InceptionV1_slim.layers = _layers_from_list_of_dicts(InceptionV1_slim(), [
  {'tags': ['conv'], 'name': 'InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu', 'depth': 64},
  {'tags': ['conv'], 'name': 'InceptionV1/InceptionV1/Conv2d_2b_1x1/Relu', 'depth': 64},
  {'tags': ['conv'], 'name': 'InceptionV1/InceptionV1/Conv2d_2c_3x3/Relu', 'depth': 192},
  {'tags': ['conv'], 'name': 'InceptionV1/InceptionV1/Mixed_3b/concat', 'depth': 256},
  {'tags': ['conv'], 'name': 'InceptionV1/InceptionV1/Mixed_3c/concat', 'depth': 480},
  {'tags': ['conv'], 'name': 'InceptionV1/InceptionV1/Mixed_4b/concat', 'depth': 512},
  {'tags': ['conv'], 'name': 'InceptionV1/InceptionV1/Mixed_4c/concat', 'depth': 512},
  {'tags': ['conv'], 'name': 'InceptionV1/InceptionV1/Mixed_4d/concat', 'depth': 512},
  {'tags': ['conv'], 'name': 'InceptionV1/InceptionV1/Mixed_4e/concat', 'depth': 528},
  {'tags': ['conv'], 'name': 'InceptionV1/InceptionV1/Mixed_4f/concat', 'depth': 832},
  {'tags': ['conv'], 'name': 'InceptionV1/InceptionV1/Mixed_5b/concat', 'depth': 832},
  {'tags': ['conv'], 'name': 'InceptionV1/InceptionV1/Mixed_5c/concat', 'depth': 1024},
  {'tags': ['dense'], 'name': 'InceptionV1/Logits/Predictions/Softmax', 'depth': 1001},
])

class InceptionV2_slim(Model):
  """InceptionV2 as implemented by the TensorFlow slim framework.

  InceptionV2 was introduced by Ioffe & Szegedy (2015):
  https://arxiv.org/pdf/1502.03167.pdf
  This function provides the pre-trained reimplementation from TF slim:
  https://github.com/tensorflow/models/tree/master/research/slim
  corresponding to the name "inception_v2".
  """

  model_path  = 'gs://modelzoo/vision/slim_models/InceptionV2.pb'
  labels_path = 'gs://modelzoo/labels/ImageNet_standard_with_dummy.txt'
  synsets_path = 'gs://modelzoo/labels/ImageNet_standard_with_dummy_synsets.txt'
  dataset = 'ImageNet'
  image_shape = [224, 224, 3]
  # inpute range taken from:
  # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py#L280
  image_value_range = (-1, 1)
  input_name = 'input'

InceptionV2_slim.layers = _layers_from_list_of_dicts(InceptionV2_slim(), [
  {'tags': ['conv'], 'name': 'InceptionV2/InceptionV2/Conv2d_1a_7x7/Relu', 'depth': 64},
  {'tags': ['conv'], 'name': 'InceptionV2/InceptionV2/Conv2d_2b_1x1/Relu', 'depth': 64},
  {'tags': ['conv'], 'name': 'InceptionV2/InceptionV2/Conv2d_2c_3x3/Relu', 'depth': 192},
  {'tags': ['conv'], 'name': 'InceptionV2/InceptionV2/Mixed_3b/concat', 'depth': 256},
  {'tags': ['conv'], 'name': 'InceptionV2/InceptionV2/Mixed_3c/concat', 'depth': 320},
  {'tags': ['conv'], 'name': 'InceptionV2/InceptionV2/Mixed_4a/concat', 'depth': 576},
  {'tags': ['conv'], 'name': 'InceptionV2/InceptionV2/Mixed_4b/concat', 'depth': 576},
  {'tags': ['conv'], 'name': 'InceptionV2/InceptionV2/Mixed_4c/concat', 'depth': 576},
  {'tags': ['conv'], 'name': 'InceptionV2/InceptionV2/Mixed_4d/concat', 'depth': 576},
  {'tags': ['conv'], 'name': 'InceptionV2/InceptionV2/Mixed_4e/concat', 'depth': 576},
  {'tags': ['conv'], 'name': 'InceptionV2/InceptionV2/Mixed_5a/concat', 'depth': 1024},
  {'tags': ['conv'], 'name': 'InceptionV2/InceptionV2/Mixed_5b/concat', 'depth': 1024},
  {'tags': ['conv'], 'name': 'InceptionV2/InceptionV2/Mixed_5c/concat', 'depth': 1024},
  {'tags': ['dense'], 'name': 'InceptionV2/Predictions/Softmax', 'depth': 1001},
])


class InceptionV3_slim(Model):
  """InceptionV3 as implemented by the TensorFlow slim framework.

  InceptionV3 was introduced by Szegedy, et al (2015)
  https://arxiv.org/pdf/1512.00567.pdf
  This function provides the pre-trained reimplementation from TF slim:
  https://github.com/tensorflow/models/tree/master/research/slim
  corresponding to the name "inception_v3".
  """

  model_path  = 'gs://modelzoo/vision/slim_models/InceptionV3.pb'
  labels_path = 'gs://modelzoo/labels/ImageNet_standard_with_dummy.txt'
  synsets_path = 'gs://modelzoo/labels/ImageNet_standard_with_dummy_synsets.txt'
  dataset = 'ImageNet'
  image_shape = [299, 299, 3]
  # inpute range taken from:
  # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py#L280
  image_value_range = (-1, 1)
  input_name = 'input'

InceptionV3_slim.layers = _layers_from_list_of_dicts(InceptionV3_slim(), [
  {'tags': ['conv'], 'name': 'InceptionV3/InceptionV3/Conv2d_1a_3x3/Relu', 'depth': 32},
  {'tags': ['conv'], 'name': 'InceptionV3/InceptionV3/Conv2d_2a_3x3/Relu', 'depth': 32},
  {'tags': ['conv'], 'name': 'InceptionV3/InceptionV3/Conv2d_2b_3x3/Relu', 'depth': 64},
  {'tags': ['conv'], 'name': 'InceptionV3/InceptionV3/Conv2d_3b_1x1/Relu', 'depth': 80},
  {'tags': ['conv'], 'name': 'InceptionV3/InceptionV3/Conv2d_4a_3x3/Relu', 'depth': 192},
  {'tags': ['conv'], 'name': 'InceptionV3/InceptionV3/Mixed_5b/concat', 'depth': 256},
  {'tags': ['conv'], 'name': 'InceptionV3/InceptionV3/Mixed_5c/concat', 'depth': 288},
  {'tags': ['conv'], 'name': 'InceptionV3/InceptionV3/Mixed_5d/concat', 'depth': 288},
  {'tags': ['conv'], 'name': 'InceptionV3/InceptionV3/Mixed_6a/concat', 'depth': 768},
  {'tags': ['conv'], 'name': 'InceptionV3/InceptionV3/Mixed_6b/concat', 'depth': 768},
  {'tags': ['conv'], 'name': 'InceptionV3/InceptionV3/Mixed_6c/concat', 'depth': 768},
  {'tags': ['conv'], 'name': 'InceptionV3/InceptionV3/Mixed_6d/concat', 'depth': 768},
  {'tags': ['conv'], 'name': 'InceptionV3/InceptionV3/Mixed_6e/concat', 'depth': 768},
  {'tags': ['conv'], 'name': 'InceptionV3/InceptionV3/Mixed_7a/concat', 'depth': 1280},
  {'tags': ['conv'], 'name': 'InceptionV3/InceptionV3/Mixed_7b/concat', 'depth': 2048},
  {'tags': ['conv'], 'name': 'InceptionV3/InceptionV3/Mixed_7c/concat', 'depth': 2048},
  {'tags': ['dense'], 'name': 'InceptionV3/Predictions/Softmax', 'depth': 1001},
])


class InceptionV4_slim(Model):
  """InceptionV4 as implemented by the TensorFlow slim framework.

  InceptionV4 was introduced by Szegedy, et al (2016):
  https://arxiv.org/pdf/1602.07261.pdf
  This function provides the pre-trained reimplementation from TF slim:
  https://github.com/tensorflow/models/tree/master/research/slim
  corresponding to the name "inception_v4".
  """

  model_path  = 'gs://modelzoo/vision/slim_models/InceptionV4.pb'
  labels_path = 'gs://modelzoo/labels/ImageNet_standard_with_dummy.txt'
  synsets_path = 'gs://modelzoo/labels/ImageNet_standard_with_dummy_synsets.txt'
  dataset = 'ImageNet'
  image_shape = [299, 299, 3]
  # inpute range taken from:
  # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py#L280
  image_value_range = (-1, 1)
  input_name = 'input'

InceptionV4_slim.layers = _layers_from_list_of_dicts(InceptionV4_slim(), [
  {'tags': ['conv'], 'name': 'InceptionV4/InceptionV4/Conv2d_1a_3x3/Relu', 'depth': 32},
  {'tags': ['conv'], 'name': 'InceptionV4/InceptionV4/Conv2d_2a_3x3/Relu', 'depth': 32},
  {'tags': ['conv'], 'name': 'InceptionV4/InceptionV4/Conv2d_2b_3x3/Relu', 'depth': 64},
  {'tags': ['conv'], 'name': 'InceptionV4/InceptionV4/Mixed_3a/concat', 'depth': 160},
  {'tags': ['conv'], 'name': 'InceptionV4/InceptionV4/Mixed_4a/concat', 'depth': 192},
  {'tags': ['conv'], 'name': 'InceptionV4/InceptionV4/Mixed_5a/concat', 'depth': 384},
  {'tags': ['conv'], 'name': 'InceptionV4/InceptionV4/Mixed_5b/concat', 'depth': 384},
  {'tags': ['conv'], 'name': 'InceptionV4/InceptionV4/Mixed_5c/concat', 'depth': 384},
  {'tags': ['conv'], 'name': 'InceptionV4/InceptionV4/Mixed_5d/concat', 'depth': 384},
  {'tags': ['conv'], 'name': 'InceptionV4/InceptionV4/Mixed_5e/concat', 'depth': 384},
  {'tags': ['conv'], 'name': 'InceptionV4/InceptionV4/Mixed_6a/concat', 'depth': 1024},
  {'tags': ['conv'], 'name': 'InceptionV4/InceptionV4/Mixed_6b/concat', 'depth': 1024},
  {'tags': ['conv'], 'name': 'InceptionV4/InceptionV4/Mixed_6c/concat', 'depth': 1024},
  {'tags': ['conv'], 'name': 'InceptionV4/InceptionV4/Mixed_6d/concat', 'depth': 1024},
  {'tags': ['conv'], 'name': 'InceptionV4/InceptionV4/Mixed_6e/concat', 'depth': 1024},
  {'tags': ['conv'], 'name': 'InceptionV4/InceptionV4/Mixed_6f/concat', 'depth': 1024},
  {'tags': ['conv'], 'name': 'InceptionV4/InceptionV4/Mixed_6g/concat', 'depth': 1024},
  {'tags': ['conv'], 'name': 'InceptionV4/InceptionV4/Mixed_6h/concat', 'depth': 1024},
  {'tags': ['conv'], 'name': 'InceptionV4/InceptionV4/Mixed_7a/concat', 'depth': 1536},
  {'tags': ['conv'], 'name': 'InceptionV4/InceptionV4/Mixed_7b/concat', 'depth': 1536},
  {'tags': ['conv'], 'name': 'InceptionV4/InceptionV4/Mixed_7c/concat', 'depth': 1536},
  {'tags': ['conv'], 'name': 'InceptionV4/InceptionV4/Mixed_7d/concat', 'depth': 1536},
  {'tags': ['dense'], 'name': 'InceptionV4/Logits/Predictions', 'depth': 1001},
])


class InceptionResnetV2_slim(Model):
  """InceptionResnetV2 as implemented by the TensorFlow slim framework.

  InceptionResnetV2 was introduced in this paper by Szegedy, et al (2016):
  https://arxiv.org/pdf/1602.07261.pdf
  This function provides the pre-trained reimplementation from TF slim:
  https://github.com/tensorflow/models/tree/master/research/slim
  corresponding to the name "inception_resnet_v2".
  """

  model_path  = 'gs://modelzoo/vision/slim_models/InceptionResnetV2.pb'
  labels_path = 'gs://modelzoo/labels/ImageNet_standard_with_dummy.txt'
  synsets_path = 'gs://modelzoo/labels/ImageNet_standard_with_dummy_synsets.txt'
  dataset = 'ImageNet'
  image_shape = [299, 299, 3]
  # inpute range taken from:
  # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py#L280
  image_value_range = (-1, 1)
  input_name = 'input'

# TODO: understand this graph, see if we can delete some add or relu nodes from layers
InceptionResnetV2_slim.layers = _layers_from_list_of_dicts(InceptionResnetV2_slim(), [
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Conv2d_1a_3x3/Relu', 'depth': 32},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Conv2d_2a_3x3/Relu', 'depth': 32},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Conv2d_2b_3x3/Relu', 'depth': 64},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Conv2d_3b_1x1/Relu', 'depth': 80},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu', 'depth': 192},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Mixed_5b/concat', 'depth': 320},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_1/add', 'depth': 320},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_1/Relu', 'depth': 320},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_2/add', 'depth': 320},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_2/Relu', 'depth': 320},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_3/add', 'depth': 320},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_3/Relu', 'depth': 320},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_4/add', 'depth': 320},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_4/Relu', 'depth': 320},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_5/add', 'depth': 320},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_5/Relu', 'depth': 320},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_6/add', 'depth': 320},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_6/Relu', 'depth': 320},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_7/add', 'depth': 320},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_7/Relu', 'depth': 320},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_8/add', 'depth': 320},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_8/Relu', 'depth': 320},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_9/add', 'depth': 320},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_9/Relu', 'depth': 320},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_10/add', 'depth': 320},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_10/Relu', 'depth': 320},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Mixed_6a/concat', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_1/add', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_1/Relu', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_2/add', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_2/Relu', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_3/add', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_3/Relu', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_4/add', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_4/Relu', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_5/add', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_5/Relu', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_6/add', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_6/Relu', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_7/add', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_7/Relu', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_8/add', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_8/Relu', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_9/add', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_9/Relu', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_10/add', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_10/Relu', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_11/add', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_11/Relu', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_12/add', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_12/Relu', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_13/add', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_13/Relu', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_14/add', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_14/Relu', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_15/add', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_15/Relu', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_16/add', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_16/Relu', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_17/add', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_17/Relu', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_18/add', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_18/Relu', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_19/add', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_19/Relu', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_20/add', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_20/Relu', 'depth': 1088},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Mixed_7a/concat', 'depth': 2080},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_1/add', 'depth': 2080},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_1/Relu', 'depth': 2080},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_2/add', 'depth': 2080},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_2/Relu', 'depth': 2080},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_3/add', 'depth': 2080},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_3/Relu', 'depth': 2080},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_4/add', 'depth': 2080},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_4/Relu', 'depth': 2080},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_5/add', 'depth': 2080},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_5/Relu', 'depth': 2080},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_6/add', 'depth': 2080},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_6/Relu', 'depth': 2080},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_7/add', 'depth': 2080},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_7/Relu', 'depth': 2080},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_8/add', 'depth': 2080},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_8/Relu', 'depth': 2080},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_9/add', 'depth': 2080},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_9/Relu', 'depth': 2080},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Block8/add', 'depth': 2080},
  {'tags': ['conv'], 'name': 'InceptionResnetV2/InceptionResnetV2/Conv2d_7b_1x1/Relu', 'depth': 1536},
  {'tags': ['dense'], 'name': 'InceptionResnetV2/Logits/Predictions', 'depth': 1001},
])
