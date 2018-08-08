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
from lucid.modelzoo.vision_base import Model, IMAGENET_MEAN_BGR


class AlexNet_caffe(Model):
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
  model_path  = 'gs://modelzoo/AlexNet.pb'
  labels_path = 'gs://modelzoo/labels/ImageNet_standard.txt'
  dataset = 'ImageNet'
  image_shape = [227, 227, 3]
  is_BGR = True
  image_value_range = (-IMAGENET_MEAN_BGR, 255-IMAGENET_MEAN_BGR)
  input_name = 'Placeholder'

  # TODO - Sanity check this graph and layers
  layers = [
     {'type': 'conv', 'name': 'concat_2', 'size': 256},
     {'type': 'conv', 'name': 'conv5_1', 'size': 256},
     {'type': 'dense', 'name': 'Relu', 'size': 4096},
     {'type': 'dense', 'name': 'Relu_1', 'size': 4096},
     {'type': 'dense', 'name': 'Softmax', 'size': 1000},
   ]


class AlexNet_caffe_Places365(Model):
  """AlexNet re-implementation trained on Places365.
  
  This model is a reimplementation of AlexNet
  https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
  trained on the MIT Places365 dataset, retrieved here:
  https://github.com/CSAILVision/places365
  and then ported to TensorFlow using caffe-tensorflow.
  """

  model_path  = 'gs://modelzoo/AlexNet_caffe_places365.pb'
  labels_path = 'gs://modelzoo/labels/Places365.txt'
  dataset = 'Places365'
  image_shape = [227, 227, 3]
  is_BGR = True
  image_value_range = (-IMAGENET_MEAN_BGR, 255-IMAGENET_MEAN_BGR)
  input_name = 'input'

  # TODO - Sanity check this graph and layers
  layers = [
    {'type': 'conv', 'name': 'conv5/concat', 'size': 256} ,
    {'type': 'conv', 'name': 'conv5/conv5', 'size': 256} ,
    {'type': 'dense', 'name': 'fc6/fc6', 'size': 4096} ,
    {'type': 'dense', 'name': 'fc7/fc7', 'size': 4096} ,
    {'type': 'dense', 'name': 'prob', 'size': 365} ,
   ]
