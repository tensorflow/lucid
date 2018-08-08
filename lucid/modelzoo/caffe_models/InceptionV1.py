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


class InceptionV1_caffe(Model):
  """InceptionV1 (or 'GoogLeNet') as reimplemented in caffe.

  This model is a reimplementation of GoogLeNet:
  https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
  reimplemented in caffe by BVLC / Sergio Guadarrama:
  https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
  and then ported to TensorFlow using caffe-tensorflow.
  """
  model_path = 'gs://modelzoo/vision/caffe_models/InceptionV1.pb'
  labels_path = 'gs://modelzoo/labels/ImageNet_standard.txt'
  dataset = 'ImageNet'
  image_shape = [224, 224, 3]
  is_BGR = True
  image_value_range = (-IMAGENET_MEAN_BGR, 255-IMAGENET_MEAN_BGR)
  input_name = 'data'

  layers = [
     {'type': 'conv', 'name': 'conv1_7x7_s2/conv1_7x7_s2', 'size': 64},
     {'type': 'conv', 'name': 'conv2_3x3_reduce/conv2_3x3_reduce', 'size': 64},
     {'type': 'conv', 'name': 'conv2_3x3/conv2_3x3', 'size': 192},
     {'type': 'conv', 'name': 'inception_3a_output', 'size': 256},
     {'type': 'conv', 'name': 'inception_3b_output', 'size': 480},
     {'type': 'conv', 'name': 'inception_4a_output', 'size': 512},
     {'type': 'conv', 'name': 'inception_4b_output', 'size': 512},
     {'type': 'conv', 'name': 'inception_4c_output', 'size': 512},
     {'type': 'conv', 'name': 'inception_4d_output', 'size': 528},
     {'type': 'conv', 'name': 'inception_4e_output', 'size': 832},
     {'type': 'conv', 'name': 'inception_5a_output', 'size': 832},
     {'type': 'conv', 'name': 'inception_5b_output', 'size': 1024},
     {'type': 'dense', 'name': 'prob', 'size': 1000},
   ]


class InceptionV1_caffe_Places205(Model):
  """InceptionV1 (or 'GoogLeNet') trained on Places205.

  This model is a caffe reimplementation of GoogLeNet:
  https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
  trained on the MIT Places205 dataset, retrieved here:
  http://places.csail.mit.edu/downloadCNN.html
  and then ported to TensorFlow using caffe-tensorflow.
  """
  model_path = 'gs://modelzoo/vision/caffe_models/InceptionV1_places205.pb'
  labels_path = 'gs://modelzoo/labels/Places205.txt'
  dataset = 'Places205'
  image_shape = [224, 224, 3]
  is_BGR = True
  image_value_range = (-IMAGENET_MEAN_BGR, 255-IMAGENET_MEAN_BGR)
  input_name = 'data'

  layers = [
     {'type': 'conv', 'name': 'conv1_7x7_s2/conv1_7x7_s2', 'size': 64},
     {'type': 'conv', 'name': 'conv2_3x3_reduce/conv2_3x3_reduce', 'size': 64},
     {'type': 'conv', 'name': 'conv2_3x3/conv2_3x3', 'size': 192},
     {'type': 'conv', 'name': 'inception_3a_output', 'size': 256},
     {'type': 'conv', 'name': 'inception_3b_output', 'size': 480},
     {'type': 'conv', 'name': 'inception_4a_output', 'size': 512},
     {'type': 'conv', 'name': 'inception_4b_output', 'size': 512},
     {'type': 'conv', 'name': 'inception_4c_output', 'size': 512},
     {'type': 'conv', 'name': 'inception_4d_output', 'size': 528},
     {'type': 'conv', 'name': 'inception_4e_output', 'size': 832},
     {'type': 'conv', 'name': 'inception_5a_output', 'size': 832},
     {'type': 'conv', 'name': 'inception_5b_output', 'size': 1024},
     {'type': 'dense', 'name': 'prob', 'size': 205},
   ]


class InceptionV1_caffe_Places365(Model):
  """InceptionV1 (or 'GoogLeNet') trained on Places365.

  This model is a caffe reimplementation of GoogLeNet:
  https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
  trained on the MIT Places365 dataset, retrieved here:
  https://github.com/CSAILVision/places365
  and then ported to TensorFlow using caffe-tensorflow.
  """
  model_path = 'gs://modelzoo/vision/caffe_models/InceptionV1_places365.pb'
  labels_path = 'gs://modelzoo/labels/Places365.txt'
  dataset = 'Places365'
  image_shape = [224, 224, 3]
  # What is the correct input range???
  is_BGR = True
  image_value_range = (-IMAGENET_MEAN_BGR, 255-IMAGENET_MEAN_BGR)
  input_name = 'data'

  layers = [
     {'type': 'conv', 'name': 'conv1_7x7_s2/conv1_7x7_s2', 'size': 64},
     {'type': 'conv', 'name': 'conv2_3x3_reduce/conv2_3x3_reduce', 'size': 64},
     {'type': 'conv', 'name': 'conv2_3x3/conv2_3x3', 'size': 192},
     {'type': 'conv', 'name': 'inception_3a_output', 'size': 256},
     {'type': 'conv', 'name': 'inception_3b_output', 'size': 480},
     {'type': 'conv', 'name': 'inception_4a_output', 'size': 512},
     {'type': 'conv', 'name': 'inception_4b_output', 'size': 512},
     {'type': 'conv', 'name': 'inception_4c_output', 'size': 512},
     {'type': 'conv', 'name': 'inception_4d_output', 'size': 528},
     {'type': 'conv', 'name': 'inception_4e_output', 'size': 832},
     {'type': 'conv', 'name': 'inception_5a_output', 'size': 832},
     {'type': 'conv', 'name': 'inception_5b_output', 'size': 1024},
     {'type': 'dense', 'name': 'prob', 'size': 365},
   ]
