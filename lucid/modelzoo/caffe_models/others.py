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
  """

  # The authors of code to convert AlexNet to TF host weights at
  # http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/alexnet_frozen.pb
  # but it seems more polite and reliable to host our own.
  model_path  = 'gs://modelzoo/AlexNet.pb'
  labels_path = 'gs://modelzoo/ImageNet_labels_caffe.txt'
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


class InceptionV1_caffe(Model):
  """InceptionV1 (or 'GoogLeNet') as reimplemented in caffe.

  This model is a reimplementation of GoogLeNet:
  https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
  reimplemented in caffe by BVLC / Sergio Guadarrama:
  https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
  and then ported to TensorFlow using caffe-tensorflow.
  """
  model_path = 'gs://modelzoo/InceptionV1_caffe.pb'
  labels_path = 'gs://modelzoo/InceptionV1_caffe-labels.txt'
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
  model_path = 'gs://modelzoo/InceptionV1_caffe_places205.pb'
  labels_path = 'gs://modelzoo/InceptionV1_caffe_places205-labels.txt'
  image_shape = [224, 224, 3]
  # range based on emperical testing
  image_value_range = (-1,1)
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
  model_path = 'gs://modelzoo/InceptionV1_caffe_places365.pb'
  # TODO - check labels match predictions
  labels_path = 'gs://modelzoo/InceptionV1_caffe_places365-labels.txt'
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
  model_path = 'gs://modelzoo/VGG16_caffe.pb'
  labels_path = 'gs://modelzoo/InceptionV1_caffe-labels.txt'
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
  model_path = 'gs://modelzoo/VGG19_caffe.pb'
  labels_path = 'gs://modelzoo/InceptionV1_caffe-labels.txt'
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
