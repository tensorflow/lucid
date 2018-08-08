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
from lucid.modelzoo.vision_base import Model, IMAGENET_MEAN


class InceptionV1_slim(Model):
  """InceptionV1 as implemented by the TensorFlow slim framework.

  InceptionV1 was introduced by Szegedy, et al (2014):
  https://arxiv.org/pdf/1409.4842v1.pdf
  This function provides the pre-trained reimplementation from TF slim:
  https://github.com/tensorflow/models/tree/master/research/slim
  corresponding to the name "inception_v1".
  """

  model_path  = 'gs://modelzoo/vision/slim_models/InceptionV1.pb'
  labels_path = 'gs://modelzoo/labels/ImageNet_standard_with_dummy.txt'
  dataset = 'ImageNet'
  image_shape = [224, 224, 3]
  # inpute range taken from:
  # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py#L280
  image_value_range = (-1, 1)
  input_name = 'input'

  layers = [
     {'type': 'conv', 'name': 'InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu', 'size': 64},
     {'type': 'conv', 'name': 'InceptionV1/InceptionV1/Conv2d_2b_1x1/Relu', 'size': 64},
     {'type': 'conv', 'name': 'InceptionV1/InceptionV1/Conv2d_2c_3x3/Relu', 'size': 192},
     {'type': 'conv', 'name': 'InceptionV1/InceptionV1/Mixed_3b/concat', 'size': 256},
     {'type': 'conv', 'name': 'InceptionV1/InceptionV1/Mixed_3c/concat', 'size': 480},
     {'type': 'conv', 'name': 'InceptionV1/InceptionV1/Mixed_4b/concat', 'size': 512},
     {'type': 'conv', 'name': 'InceptionV1/InceptionV1/Mixed_4c/concat', 'size': 512},
     {'type': 'conv', 'name': 'InceptionV1/InceptionV1/Mixed_4d/concat', 'size': 512},
     {'type': 'conv', 'name': 'InceptionV1/InceptionV1/Mixed_4e/concat', 'size': 528},
     {'type': 'conv', 'name': 'InceptionV1/InceptionV1/Mixed_4f/concat', 'size': 832},
     {'type': 'conv', 'name': 'InceptionV1/InceptionV1/Mixed_5b/concat', 'size': 832},
     {'type': 'conv', 'name': 'InceptionV1/InceptionV1/Mixed_5c/concat', 'size': 1024},
     {'type': 'dense', 'name': 'InceptionV1/Logits/Predictions/Softmax', 'size': 1001},
   ]


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
  dataset = 'ImageNet'
  image_shape = [224, 224, 3]
  # inpute range taken from:
  # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py#L280
  image_value_range = (-1, 1)
  input_name = 'input'

  layers = [
     {'type': 'conv', 'name': 'InceptionV2/InceptionV2/Conv2d_1a_7x7/Relu', 'size': 64},
     {'type': 'conv', 'name': 'InceptionV2/InceptionV2/Conv2d_2b_1x1/Relu', 'size': 64},
     {'type': 'conv', 'name': 'InceptionV2/InceptionV2/Conv2d_2c_3x3/Relu', 'size': 192},
     {'type': 'conv', 'name': 'InceptionV2/InceptionV2/Mixed_3b/concat', 'size': 256},
     {'type': 'conv', 'name': 'InceptionV2/InceptionV2/Mixed_3c/concat', 'size': 320},
     {'type': 'conv', 'name': 'InceptionV2/InceptionV2/Mixed_4a/concat', 'size': 576},
     {'type': 'conv', 'name': 'InceptionV2/InceptionV2/Mixed_4b/concat', 'size': 576},
     {'type': 'conv', 'name': 'InceptionV2/InceptionV2/Mixed_4c/concat', 'size': 576},
     {'type': 'conv', 'name': 'InceptionV2/InceptionV2/Mixed_4d/concat', 'size': 576},
     {'type': 'conv', 'name': 'InceptionV2/InceptionV2/Mixed_4e/concat', 'size': 576},
     {'type': 'conv', 'name': 'InceptionV2/InceptionV2/Mixed_5a/concat', 'size': 1024},
     {'type': 'conv', 'name': 'InceptionV2/InceptionV2/Mixed_5b/concat', 'size': 1024},
     {'type': 'conv', 'name': 'InceptionV2/InceptionV2/Mixed_5c/concat', 'size': 1024},
     {'type': 'dense', 'name': 'InceptionV2/Predictions/Softmax', 'size': 1001},
   ]


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
  dataset = 'ImageNet'
  image_shape = [299, 299, 3]
  # inpute range taken from:
  # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py#L280
  image_value_range = (-1, 1)
  input_name = 'input'

  layers = [
     {'type': 'conv', 'name': 'InceptionV3/InceptionV3/Conv2d_1a_3x3/Relu', 'size': 32},
     {'type': 'conv', 'name': 'InceptionV3/InceptionV3/Conv2d_2a_3x3/Relu', 'size': 32},
     {'type': 'conv', 'name': 'InceptionV3/InceptionV3/Conv2d_2b_3x3/Relu', 'size': 64},
     {'type': 'conv', 'name': 'InceptionV3/InceptionV3/Conv2d_3b_1x1/Relu', 'size': 80},
     {'type': 'conv', 'name': 'InceptionV3/InceptionV3/Conv2d_4a_3x3/Relu', 'size': 192},
     {'type': 'conv', 'name': 'InceptionV3/InceptionV3/Mixed_5b/concat', 'size': 256},
     {'type': 'conv', 'name': 'InceptionV3/InceptionV3/Mixed_5c/concat', 'size': 288},
     {'type': 'conv', 'name': 'InceptionV3/InceptionV3/Mixed_5d/concat', 'size': 288},
     {'type': 'conv', 'name': 'InceptionV3/InceptionV3/Mixed_6a/concat', 'size': 768},
     {'type': 'conv', 'name': 'InceptionV3/InceptionV3/Mixed_6b/concat', 'size': 768},
     {'type': 'conv', 'name': 'InceptionV3/InceptionV3/Mixed_6c/concat', 'size': 768},
     {'type': 'conv', 'name': 'InceptionV3/InceptionV3/Mixed_6d/concat', 'size': 768},
     {'type': 'conv', 'name': 'InceptionV3/InceptionV3/Mixed_6e/concat', 'size': 768},
     {'type': 'conv', 'name': 'InceptionV3/InceptionV3/Mixed_7a/concat', 'size': 1280},
     {'type': 'conv', 'name': 'InceptionV3/InceptionV3/Mixed_7b/concat', 'size': 2048},
     {'type': 'conv', 'name': 'InceptionV3/InceptionV3/Mixed_7c/concat', 'size': 2048},
     {'type': 'dense', 'name': 'InceptionV3/Predictions/Softmax', 'size': 1001},
   ]


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
  dataset = 'ImageNet'
  image_shape = [299, 299, 3]
  # inpute range taken from:
  # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py#L280
  image_value_range = (-1, 1)
  input_name = 'input'

  layers = [
     {'type': 'conv', 'name': 'InceptionV4/InceptionV4/Conv2d_1a_3x3/Relu', 'size': 32},
     {'type': 'conv', 'name': 'InceptionV4/InceptionV4/Conv2d_2a_3x3/Relu', 'size': 32},
     {'type': 'conv', 'name': 'InceptionV4/InceptionV4/Conv2d_2b_3x3/Relu', 'size': 64},
     {'type': 'conv', 'name': 'InceptionV4/InceptionV4/Mixed_3a/concat', 'size': 160},
     {'type': 'conv', 'name': 'InceptionV4/InceptionV4/Mixed_4a/concat', 'size': 192},
     {'type': 'conv', 'name': 'InceptionV4/InceptionV4/Mixed_5a/concat', 'size': 384},
     {'type': 'conv', 'name': 'InceptionV4/InceptionV4/Mixed_5b/concat', 'size': 384},
     {'type': 'conv', 'name': 'InceptionV4/InceptionV4/Mixed_5c/concat', 'size': 384},
     {'type': 'conv', 'name': 'InceptionV4/InceptionV4/Mixed_5d/concat', 'size': 384},
     {'type': 'conv', 'name': 'InceptionV4/InceptionV4/Mixed_5e/concat', 'size': 384},
     {'type': 'conv', 'name': 'InceptionV4/InceptionV4/Mixed_6a/concat', 'size': 1024},
     {'type': 'conv', 'name': 'InceptionV4/InceptionV4/Mixed_6b/concat', 'size': 1024},
     {'type': 'conv', 'name': 'InceptionV4/InceptionV4/Mixed_6c/concat', 'size': 1024},
     {'type': 'conv', 'name': 'InceptionV4/InceptionV4/Mixed_6d/concat', 'size': 1024},
     {'type': 'conv', 'name': 'InceptionV4/InceptionV4/Mixed_6e/concat', 'size': 1024},
     {'type': 'conv', 'name': 'InceptionV4/InceptionV4/Mixed_6f/concat', 'size': 1024},
     {'type': 'conv', 'name': 'InceptionV4/InceptionV4/Mixed_6g/concat', 'size': 1024},
     {'type': 'conv', 'name': 'InceptionV4/InceptionV4/Mixed_6h/concat', 'size': 1024},
     {'type': 'conv', 'name': 'InceptionV4/InceptionV4/Mixed_7a/concat', 'size': 1536},
     {'type': 'conv', 'name': 'InceptionV4/InceptionV4/Mixed_7b/concat', 'size': 1536},
     {'type': 'conv', 'name': 'InceptionV4/InceptionV4/Mixed_7c/concat', 'size': 1536},
     {'type': 'conv', 'name': 'InceptionV4/InceptionV4/Mixed_7d/concat', 'size': 1536},
     {'type': 'dense', 'name': 'InceptionV4/Logits/Predictions', 'size': 1001},
   ]


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
  dataset = 'ImageNet'
  image_shape = [299, 299, 3]
  # inpute range taken from:
  # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py#L280
  image_value_range = (-1, 1)
  input_name = 'input'

  # TODO: understand this graph, see if we can delete some add or relu nodes from layers
  layers = [
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Conv2d_1a_3x3/Relu', 'size': 32},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Conv2d_2a_3x3/Relu', 'size': 32},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Conv2d_2b_3x3/Relu', 'size': 64},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Conv2d_3b_1x1/Relu', 'size': 80},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu', 'size': 192},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Mixed_5b/concat', 'size': 320},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_1/add', 'size': 320},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_1/Relu', 'size': 320},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_2/add', 'size': 320},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_2/Relu', 'size': 320},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_3/add', 'size': 320},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_3/Relu', 'size': 320},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_4/add', 'size': 320},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_4/Relu', 'size': 320},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_5/add', 'size': 320},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_5/Relu', 'size': 320},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_6/add', 'size': 320},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_6/Relu', 'size': 320},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_7/add', 'size': 320},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_7/Relu', 'size': 320},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_8/add', 'size': 320},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_8/Relu', 'size': 320},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_9/add', 'size': 320},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_9/Relu', 'size': 320},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_10/add', 'size': 320},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat/block35_10/Relu', 'size': 320},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Mixed_6a/concat', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_1/add', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_1/Relu', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_2/add', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_2/Relu', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_3/add', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_3/Relu', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_4/add', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_4/Relu', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_5/add', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_5/Relu', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_6/add', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_6/Relu', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_7/add', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_7/Relu', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_8/add', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_8/Relu', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_9/add', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_9/Relu', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_10/add', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_10/Relu', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_11/add', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_11/Relu', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_12/add', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_12/Relu', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_13/add', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_13/Relu', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_14/add', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_14/Relu', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_15/add', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_15/Relu', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_16/add', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_16/Relu', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_17/add', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_17/Relu', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_18/add', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_18/Relu', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_19/add', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_19/Relu', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_20/add', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_20/Relu', 'size': 1088},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Mixed_7a/concat', 'size': 2080},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_1/add', 'size': 2080},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_1/Relu', 'size': 2080},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_2/add', 'size': 2080},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_2/Relu', 'size': 2080},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_3/add', 'size': 2080},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_3/Relu', 'size': 2080},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_4/add', 'size': 2080},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_4/Relu', 'size': 2080},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_5/add', 'size': 2080},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_5/Relu', 'size': 2080},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_6/add', 'size': 2080},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_6/Relu', 'size': 2080},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_7/add', 'size': 2080},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_7/Relu', 'size': 2080},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_8/add', 'size': 2080},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_8/Relu', 'size': 2080},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_9/add', 'size': 2080},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Repeat_2/block8_9/Relu', 'size': 2080},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Block8/add', 'size': 2080},
     {'type': 'conv', 'name': 'InceptionResnetV2/InceptionResnetV2/Conv2d_7b_1x1/Relu', 'size': 1536},
     {'type': 'dense', 'name': 'InceptionResnetV2/Logits/Predictions', 'size': 1001},
   ]
