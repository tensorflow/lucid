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
from lucid.modelzoo.vision_base import Model, _layers_from_list_of_dicts, IMAGENET_MEAN_BGR


class AlexNet_caffe_Places365(Model):
  """AlexNet re-implementation trained on Places365.

  This model is a reimplementation of AlexNet
  https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
  trained on the MIT Places365 dataset, retrieved here:
  https://github.com/CSAILVision/places365
  and then ported to TensorFlow using caffe-tensorflow.
  """

  model_path  = 'gs://modelzoo/vision/caffe_models/AlexNet_places365.pb'
  labels_path = 'gs://modelzoo/labels/Places365.txt'
  dataset = 'Places365'
  image_shape = [227, 227, 3]
  is_BGR = True
  image_value_range = (-IMAGENET_MEAN_BGR, 255-IMAGENET_MEAN_BGR)
  input_name = 'input'

# TODO - Sanity check this graph and layers
AlexNet_caffe_Places365.layers = _layers_from_list_of_dicts(AlexNet_caffe_Places365(), [
  {'tags': ['conv'], 'name': 'conv5/concat', 'depth': 256} ,
  {'tags': ['conv'], 'name': 'conv5/conv5', 'depth': 256} ,
  {'tags': ['dense'], 'name': 'fc6/fc6', 'depth': 4096} ,
  {'tags': ['dense'], 'name': 'fc7/fc7', 'depth': 4096} ,
  {'tags': ['dense'], 'name': 'prob', 'depth': 365} ,
])
