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


class MobilenetV2_10_slim(Model):
  """MobilenetV2 1.0 as implemented by the TensorFlow slim framework.
  
  This function provides the pre-trained reimplementation from TF slim:
  https://github.com/tensorflow/models/tree/master/research/slim
  """
  
  model_path  = 'gs://modelzoo/vision/slim_models/MobilenetV2_10.pb'
  labels_path = 'gs://modelzoo/labels/ImageNet_standard_with_dummy.txt'
  dataset = 'ImageNet'
  image_shape = [224, 224, 3]
  # inpute range taken from:
  # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py#L280
  image_value_range = (-1, 1)
  input_name = 'input'
  layers = [
     {'type': 'conv', 'name': 'MobilenetV2/expanded_conv_2/add', 'size': 24} ,
     {'type': 'conv', 'name': 'MobilenetV2/expanded_conv_4/add', 'size': 32} ,
     {'type': 'conv', 'name': 'MobilenetV2/expanded_conv_5/add', 'size': 32} ,
     {'type': 'conv', 'name': 'MobilenetV2/expanded_conv_7/add', 'size': 64} ,
     {'type': 'conv', 'name': 'MobilenetV2/expanded_conv_8/add', 'size': 64} ,
     {'type': 'conv', 'name': 'MobilenetV2/expanded_conv_9/add', 'size': 64} ,
     {'type': 'conv', 'name': 'MobilenetV2/expanded_conv_11/add', 'size': 96} ,
     {'type': 'conv', 'name': 'MobilenetV2/expanded_conv_12/add', 'size': 96} ,
     {'type': 'conv', 'name': 'MobilenetV2/expanded_conv_14/add', 'size': 160} ,
     {'type': 'conv', 'name': 'MobilenetV2/expanded_conv_15/add', 'size': 160} ,
     {'type': 'dense', 'name': 'MobilenetV2/Predictions/Softmax', 'size': 1001} ,
   ]


class MobilenetV2_14_slim(Model):
  """MobilenetV2 1.4 as implemented by the TensorFlow slim framework.
  
  This function provides the pre-trained reimplementation from TF slim:
  https://github.com/tensorflow/models/tree/master/research/slim
  """
  
  model_path  = 'gs://modelzoo/vision/slim_models/MobilenetV2_14.pb'
  labels_path = 'gs://modelzoo/labels/ImageNet_standard_with_dummy.txt'
  dataset = 'ImageNet'
  image_shape = [224, 224, 3]
  # inpute range taken from:
  # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py#L280
  image_value_range = (-1, 1)
  input_name = 'input'
  
  layers = [
     {'type': 'conv', 'name': 'MobilenetV2/expanded_conv_2/add', 'size': 32} ,
     {'type': 'conv', 'name': 'MobilenetV2/expanded_conv_4/add', 'size': 48} ,
     {'type': 'conv', 'name': 'MobilenetV2/expanded_conv_5/add', 'size': 48} ,
     {'type': 'conv', 'name': 'MobilenetV2/expanded_conv_7/add', 'size': 88} ,
     {'type': 'conv', 'name': 'MobilenetV2/expanded_conv_8/add', 'size': 88} ,
     {'type': 'conv', 'name': 'MobilenetV2/expanded_conv_9/add', 'size': 88} ,
     {'type': 'conv', 'name': 'MobilenetV2/expanded_conv_11/add', 'size': 136} ,
     {'type': 'conv', 'name': 'MobilenetV2/expanded_conv_12/add', 'size': 136} ,
     {'type': 'conv', 'name': 'MobilenetV2/expanded_conv_14/add', 'size': 224} ,
     {'type': 'conv', 'name': 'MobilenetV2/expanded_conv_15/add', 'size': 224} ,
     {'type': 'dense', 'name': 'MobilenetV2/Predictions/Softmax', 'size': 1001} ,
   ]
  
  
