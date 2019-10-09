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


class AlexNet(Model):
  """Original AlexNet weights ported to TF.

  AlexNet is the breakthrough vision model from Krizhevsky, et al (2012):
  https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
  This implementation is a caffe re-implementation:
  http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
  It was converted to TensorFlow by this GitHub project:
  https://github.com/huanzhang12/tensorflow-alexnet-model
  It appears the parameters are the actual original parameters.
  """

  # The authors of code to convert AlexNet to TF host weights at
  # http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/alexnet_frozen.pb
  # but it seems more polite and reliable to host our own.
  model_path  = 'gs://modelzoo/vision/other_models/AlexNet.pb'
  labels_path = 'gs://modelzoo/labels/ImageNet_standard.txt'
  synsets_path = 'gs://modelzoo/labels/ImageNet_standard_synsets.txt'
  dataset = 'ImageNet'
  image_shape = [227, 227, 3]
  is_BGR = True
  image_value_range = (-IMAGENET_MEAN_BGR, 255-IMAGENET_MEAN_BGR)
  input_name = 'Placeholder'

AlexNet.layers = _layers_from_list_of_dicts(AlexNet(), [
  {'tags': ['pre_relu', 'conv'], 'name': 'Conv2D', 'depth': 96},
  {'tags': ['pre_relu', 'conv'], 'name': 'Conv2D_1', 'depth': 128},
  {'tags': ['pre_relu', 'conv'], 'name': 'Conv2D_2', 'depth': 128},
  {'tags': ['pre_relu', 'conv'], 'name': 'Conv2D_3', 'depth': 384},
  {'tags': ['pre_relu', 'conv'], 'name': 'Conv2D_4', 'depth': 192},
  {'tags': ['pre_relu', 'conv'], 'name': 'Conv2D_5', 'depth': 192},
  {'tags': ['pre_relu', 'conv'], 'name': 'Conv2D_6', 'depth': 128},
  {'tags': ['pre_relu', 'conv'], 'name': 'Conv2D_7', 'depth': 128},
  {'tags': ['dense'], 'name': 'Relu', 'depth': 4096},
  {'tags': ['dense'], 'name': 'Relu_1', 'depth': 4096},
  {'tags': ['dense'], 'name': 'Softmax', 'depth': 1000},
])
