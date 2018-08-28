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

from __future__ import absolute_import, division, print_function
from lucid.modelzoo.vision_base import Model


class MobilenetV1_slim(Model):
  """MobilenetV1 as implemented by the TensorFlow slim framework.

  This function provides the pre-trained reimplementation from TF slim:
  https://github.com/tensorflow/models/tree/master/research/slim
  """

  model_path  = 'gs://modelzoo/vision/slim_models/MobilenetV1.pb'
  labels_path = 'gs://modelzoo/labels/ImageNet_standard_with_dummy.txt' #TODO
  dataset = 'ImageNet'
  image_shape = [224, 224, 3]
  # inpute range taken from:
  # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py#L280
  image_value_range = (-1, 1)
  input_name = 'input'
  layers = [
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_0/Relu6', 'size': 32, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6', 'size': 64, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6', 'size': 128, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6', 'size': 128, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6', 'size': 256, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6', 'size': 256, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6', 'size': 512, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6', 'size': 512, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6', 'size': 512, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6', 'size': 512, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6', 'size': 512, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6', 'size': 512, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6', 'size': 1024, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6', 'size': 1024, 'type': 'conv'},
    # {'name': u'MobilenetV1/Logits/AvgPool_1a/AvgPool', 'size': 1024, 'type': 'avgpool'},
    # {'name': u'MobilenetV1/Logits/Conv2d_1c_1x1/Conv2D', 'size': 1001, 'type': 'conv'},
    {'name': u'MobilenetV1/Predictions/Softmax', 'size': 1001, 'type': 'dense'},
  ]


class MobilenetV1_050_slim(Model):
  """MobilenetV1050 as implemented by the TensorFlow slim framework.

  This function provides the pre-trained reimplementation from TF slim:
  https://github.com/tensorflow/models/tree/master/research/slim
  """

  model_path  = 'gs://modelzoo/vision/slim_models/MobilenetV1050.pb'
  labels_path = 'gs://modelzoo/labels/ImageNet_standard_with_dummy.txt' #TODO
  dataset = 'ImageNet'
  image_shape = [224, 224, 3]
  # inpute range taken from:
  # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py#L280
  image_value_range = (-1, 1)
  input_name = 'input'
  layers = [
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_0/Relu6', 'size': 16, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6', 'size': 32, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6', 'size': 64, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6', 'size': 64, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6', 'size': 128, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6', 'size': 128, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6', 'size': 256, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6', 'size': 256, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6', 'size': 256, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6', 'size': 256, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6', 'size': 256, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6', 'size': 256, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6', 'size': 512, 'type': 'conv'},
    {'name': u'MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6', 'size': 512, 'type': 'conv'},
    # {'name': u'MobilenetV1/Logits/AvgPool_1a/AvgPool', 'size': 512, 'type': 'avgpool'},
    # {'name': u'MobilenetV1/Logits/Conv2d_1c_1x1/Conv2D', 'size': 1001, 'type': 'conv'},
    {'name': u'MobilenetV1/Predictions/Softmax', 'size': 1001, 'type': 'dense'},
  ]


class MobilenetV1_025_slim(Model):
  """MobilenetV1025 as implemented by the TensorFlow slim framework.

  This function provides the pre-trained reimplementation from TF slim:
  https://github.com/tensorflow/models/tree/master/research/slim
  """

  model_path  = 'gs://modelzoo/vision/slim_models/MobilenetV1025.pb'
  labels_path = 'gs://modelzoo/labels/ImageNet_standard_with_dummy.txt' #TODO
  dataset = 'ImageNet'
  image_shape = [224, 224, 3]
  # inpute range taken from:
  # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py#L280
  image_value_range = (-1, 1)
  input_name = 'input'
  layers = [
    {'type': 'conv', 'name': u'MobilenetV1/MobilenetV1/Conv2d_0/Relu6', 'size': 8},
    {'type': 'conv', 'name': u'MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6', 'size': 16},
    {'type': 'conv', 'name': u'MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6', 'size': 32},
    {'type': 'conv', 'name': u'MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6', 'size': 32},
    {'type': 'conv', 'name': u'MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6', 'size': 64},
    {'type': 'conv', 'name': u'MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6', 'size': 64},
    {'type': 'conv', 'name': u'MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6', 'size': 128},
    {'type': 'conv', 'name': u'MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6', 'size': 128},
    {'type': 'conv', 'name': u'MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6', 'size': 128},
    {'type': 'conv', 'name': u'MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6', 'size': 128},
    {'type': 'conv', 'name': u'MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6', 'size': 128},
    {'type': 'conv', 'name': u'MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6', 'size': 128},
    {'type': 'conv', 'name': u'MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6', 'size': 256},
    {'type': 'conv', 'name': u'MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6', 'size': 256},
    # {'type': 'avgpool', 'name': u'MobilenetV1/Logits/AvgPool_1a/AvgPool', 'size': 256},
    # {'type': 'conv', 'name': u'MobilenetV1/Logits/Conv2d_1c_1x1/Conv2D', 'size': 1001},
    {'type': 'dense', 'name': u'MobilenetV1/Predictions/Softmax', 'size': 1001},
  ]
