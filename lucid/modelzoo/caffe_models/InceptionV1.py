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
  synsets_path = 'gs://modelzoo/labels/ImageNet_standard_synsets.txt'
  dataset = 'ImageNet'
  image_shape = [224, 224, 3]
  is_BGR = True
  image_value_range = (-IMAGENET_MEAN_BGR, 255-IMAGENET_MEAN_BGR)
  input_name = 'data'

InceptionV1_caffe.layers = _layers_from_list_of_dicts(InceptionV1_caffe(), [
  {'tags': ['conv'], 'name': 'conv1_7x7_s2/conv1_7x7_s2', 'depth': 64},
  {'tags': ['conv'], 'name': 'conv2_3x3_reduce/conv2_3x3_reduce', 'depth': 64},
  {'tags': ['conv'], 'name': 'conv2_3x3/conv2_3x3', 'depth': 192},
  {'tags': ['conv'], 'name': 'inception_3a_output', 'depth': 256},
  {'tags': ['conv'], 'name': 'inception_3b_output', 'depth': 480},
  {'tags': ['conv'], 'name': 'inception_4a_output', 'depth': 512},
  {'tags': ['conv'], 'name': 'inception_4b_output', 'depth': 512},
  {'tags': ['conv'], 'name': 'inception_4c_output', 'depth': 512},
  {'tags': ['conv'], 'name': 'inception_4d_output', 'depth': 528},
  {'tags': ['conv'], 'name': 'inception_4e_output', 'depth': 832},
  {'tags': ['conv'], 'name': 'inception_5a_output', 'depth': 832},
  {'tags': ['conv'], 'name': 'inception_5b_output', 'depth': 1024},
  {'tags': ['dense'], 'name': 'prob', 'depth': 1000},
])


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

InceptionV1_caffe_Places205.layers = _layers_from_list_of_dicts(InceptionV1_caffe_Places205(), [
  {'tags': ['conv'], 'name': 'conv1_7x7_s2/conv1_7x7_s2', 'depth': 64},
  {'tags': ['conv'], 'name': 'conv2_3x3_reduce/conv2_3x3_reduce', 'depth': 64},
  {'tags': ['conv'], 'name': 'conv2_3x3/conv2_3x3', 'depth': 192},
  {'tags': ['conv'], 'name': 'inception_3a_output', 'depth': 256},
  {'tags': ['conv'], 'name': 'inception_3b_output', 'depth': 480},
  {'tags': ['conv'], 'name': 'inception_4a_output', 'depth': 512},
  {'tags': ['conv'], 'name': 'inception_4b_output', 'depth': 512},
  {'tags': ['conv'], 'name': 'inception_4c_output', 'depth': 512},
  {'tags': ['conv'], 'name': 'inception_4d_output', 'depth': 528},
  {'tags': ['conv'], 'name': 'inception_4e_output', 'depth': 832},
  {'tags': ['conv'], 'name': 'inception_5a_output', 'depth': 832},
  {'tags': ['conv'], 'name': 'inception_5b_output', 'depth': 1024},
  {'tags': ['dense'], 'name': 'prob', 'depth': 205},
])


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

InceptionV1_caffe_Places365.layers = _layers_from_list_of_dicts(InceptionV1_caffe_Places365(), [
  {'tags': ['conv'], 'name': 'conv1_7x7_s2/conv1_7x7_s2', 'depth': 64},
  {'tags': ['conv'], 'name': 'conv2_3x3_reduce/conv2_3x3_reduce', 'depth': 64},
  {'tags': ['conv'], 'name': 'conv2_3x3/conv2_3x3', 'depth': 192},
  {'tags': ['conv'], 'name': 'inception_3a_output', 'depth': 256},
  {'tags': ['conv'], 'name': 'inception_3b_output', 'depth': 480},
  {'tags': ['conv'], 'name': 'inception_4a_output', 'depth': 512},
  {'tags': ['conv'], 'name': 'inception_4b_output', 'depth': 512},
  {'tags': ['conv'], 'name': 'inception_4c_output', 'depth': 512},
  {'tags': ['conv'], 'name': 'inception_4d_output', 'depth': 528},
  {'tags': ['conv'], 'name': 'inception_4e_output', 'depth': 832},
  {'tags': ['conv'], 'name': 'inception_5a_output', 'depth': 832},
  {'tags': ['conv'], 'name': 'inception_5b_output', 'depth': 1024},
  {'tags': ['dense'], 'name': 'prob', 'depth': 365},
])
