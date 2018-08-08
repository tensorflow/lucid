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

class CaffeNet_caffe(Model):
  """CaffeNet (AlexNet variant included in Caffe)
  
  CaffeNet is a slight variant on AlexNet, described here:
  https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet
  """

  model_path  = 'gs://modelzoo/vision/caffe_models/CaffeNet.pb'
  labels_path = 'gs://modelzoo/labels/ImageNet_standard.txt'
  dataset = 'ImageNet'
  image_shape = [227, 227, 3]
  is_BGR = True
  image_value_range = (-IMAGENET_MEAN_BGR, 255-IMAGENET_MEAN_BGR)
  input_name = 'data'

  layers = [
    {'type': 'conv', 'name': 'conv5/concat', 'size': 256} ,
    {'type': 'conv', 'name': 'conv5/conv5', 'size': 256} ,
    {'type': 'dense', 'name': 'fc6/fc6', 'size': 4096} ,
    {'type': 'dense', 'name': 'fc7/fc7', 'size': 4096} ,
    {'type': 'dense', 'name': 'prob', 'size': 1000} ,
   ]


class VGG16_caffe(Model):
  """VGG16 model used in ImageNet ILSVRC-2014, ported from caffe.

  VGG16 was introduced by Simonyan & Zisserman (2014):
  https://arxiv.org/pdf/1409.1556.pdf
  http://www.robots.ox.ac.uk/~vgg/research/very_deep/
  as the Oxford Visual Geometry Group's submission for the ImageNet ILSVRC-2014
  contest. We download their caffe trained model from
  https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
  and convert it with caffe-tensorflow.
  """
  model_path = 'gs://modelzoo/vision/caffe_models/VGG16.pb'
  labels_path = 'gs://modelzoo/labels/ImageNet_standard.txt'
  dataset = 'ImageNet'
  image_shape = [224, 224, 3]
  is_BGR = True
  image_value_range = (-IMAGENET_MEAN_BGR, 255-IMAGENET_MEAN_BGR)
  input_name = 'input'

  layers = [
     {'type': 'conv', 'name': 'conv1_1/conv1_1', 'size': 64},
     {'type': 'conv', 'name': 'conv1_2/conv1_2', 'size': 64},
     {'type': 'conv', 'name': 'conv2_1/conv2_1', 'size': 128},
     {'type': 'conv', 'name': 'conv2_2/conv2_2', 'size': 128},
     {'type': 'conv', 'name': 'conv3_1/conv3_1', 'size': 256},
     {'type': 'conv', 'name': 'conv3_2/conv3_2', 'size': 256},
     {'type': 'conv', 'name': 'conv3_3/conv3_3', 'size': 256},
     {'type': 'conv', 'name': 'conv4_1/conv4_1', 'size': 512},
     {'type': 'conv', 'name': 'conv4_2/conv4_2', 'size': 512},
     {'type': 'conv', 'name': 'conv4_3/conv4_3', 'size': 512},
     {'type': 'conv', 'name': 'conv5_1/conv5_1', 'size': 512},
     {'type': 'conv', 'name': 'conv5_2/conv5_2', 'size': 512},
     {'type': 'conv', 'name': 'conv5_3/conv5_3', 'size': 512},
     {'type': 'dense', 'name': 'fc6/fc6', 'size': 4096},
     {'type': 'dense', 'name': 'fc7/fc7', 'size': 4096},
     {'type': 'dense', 'name': 'prob', 'size': 1000},
   ]


class VGG19_caffe(Model):
  """VGG16 model used in ImageNet ILSVRC-2014, ported from caffe.

  VGG19 was introduced by Simonyan & Zisserman (2014):
  https://arxiv.org/pdf/1409.1556.pdf
  http://www.robots.ox.ac.uk/~vgg/research/very_deep/
  as the Oxford Visual Geometry Group's submission for the ImageNet ILSVRC-2014
  contest. We download their caffe trained model from
  https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md
  and convert it with caffe-tensorflow.
  """
  model_path = 'gs://modelzoo/vision/caffe_models/VGG19.pb'
  labels_path = 'gs://modelzoo/labels/ImageNet_standard.txt'
  dataset = 'ImageNet'
  image_shape = [224, 224, 3]
  is_BGR = True
  image_value_range = (-IMAGENET_MEAN_BGR, 255-IMAGENET_MEAN_BGR)
  input_name = 'input'

  layers = [
     {'type': 'conv', 'name': 'conv1_1/conv1_1', 'size': 64},
     {'type': 'conv', 'name': 'conv1_2/conv1_2', 'size': 64},
     {'type': 'conv', 'name': 'conv2_1/conv2_1', 'size': 128},
     {'type': 'conv', 'name': 'conv2_2/conv2_2', 'size': 128},
     {'type': 'conv', 'name': 'conv3_1/conv3_1', 'size': 256},
     {'type': 'conv', 'name': 'conv3_2/conv3_2', 'size': 256},
     {'type': 'conv', 'name': 'conv3_3/conv3_3', 'size': 256},
     {'type': 'conv', 'name': 'conv3_4/conv3_4', 'size': 256},
     {'type': 'conv', 'name': 'conv4_1/conv4_1', 'size': 512},
     {'type': 'conv', 'name': 'conv4_2/conv4_2', 'size': 512},
     {'type': 'conv', 'name': 'conv4_3/conv4_3', 'size': 512},
     {'type': 'conv', 'name': 'conv4_4/conv4_4', 'size': 512},
     {'type': 'conv', 'name': 'conv5_1/conv5_1', 'size': 512},
     {'type': 'conv', 'name': 'conv5_2/conv5_2', 'size': 512},
     {'type': 'conv', 'name': 'conv5_3/conv5_3', 'size': 512},
     {'type': 'conv', 'name': 'conv5_4/conv5_4', 'size': 512},
     {'type': 'dense', 'name': 'fc6/fc6', 'size': 4096},
     {'type': 'dense', 'name': 'fc7/fc7', 'size': 4096},
     {'type': 'dense', 'name': 'prob', 'size': 1000},
   ]
