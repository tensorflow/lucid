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

import tensorflow as tf
import numpy as np

from lucid.modelzoo.vision_base import Model


IMAGENET_MEAN = np.array([123.68, 116.779, 103.939])
IMAGENET_MEAN_BGR = np.flip(IMAGENET_MEAN, 0)


def _populate_inception_bottlenecks(scope):
  """Add Inception bottlenecks and their pre-Relu versions to the graph."""
  graph = tf.get_default_graph()
  for op in graph.get_operations():
    if op.name.startswith(scope+'/') and 'Concat' in op.type:
      name = op.name.split('/')[1]
      pre_relus = []
      for tower in op.inputs[1:]:
        if tower.op.type == 'Relu':
          tower = tower.op.inputs[0]
        pre_relus.append(tower)
      concat_name = scope + '/' + name + '_pre_relu'
      _ = tf.concat(pre_relus, -1, name=concat_name)


class InceptionV1(Model):
  """InceptionV1 (or 'GoogLeNet')

  https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
  """
  model_path = 'gs://modelzoo/InceptionV1.pb'
  labels_path = 'gs://modelzoo/InceptionV1-labels.txt'
  image_shape = [224, 224, 3]
  image_value_range = (-117, 255-117)
  input_name = 'input:0'

  def post_import(self, scope):
    _populate_inception_bottlenecks(scope)

  layers = [
     {'type': 'conv', 'name': 'conv2d0', 'size': 64},
     {'type': 'conv', 'name': 'conv2d1', 'size': 64},
     {'type': 'conv', 'name': 'conv2d2', 'size': 192},
     {'type': 'conv', 'name': 'mixed3a', 'size': 256},
     {'type': 'conv', 'name': 'mixed3b', 'size': 480},
     {'type': 'conv', 'name': 'mixed4a', 'size': 508},
     {'type': 'conv', 'name': 'mixed4b', 'size': 512},
     {'type': 'conv', 'name': 'mixed4c', 'size': 512},
     {'type': 'conv', 'name': 'mixed4d', 'size': 528},
     {'type': 'conv', 'name': 'mixed4e', 'size': 832},
     {'type': 'conv', 'name': 'mixed5a', 'size': 832},
     {'type': 'conv', 'name': 'mixed5b', 'size': 1024},
     {'type': 'conv', 'name': 'head0_bottleneck', 'size': 128},
     {'type': 'dense', 'name': 'nn0', 'size': 1024},
     {'type': 'dense', 'name': 'softmax0', 'size': 1008},
     {'type': 'conv', 'name': 'head1_bottleneck', 'size': 128},
     {'type': 'dense', 'name': 'nn1', 'size': 1024},
     {'type': 'dense', 'name': 'softmax1', 'size': 1008},
     {'type': 'dense', 'name': 'softmax2', 'size': 1008},
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


class InceptionV1_slim(Model):
  """InceptionV1 as implemented by the TensorFlow slim framework.

  InceptionV1 was introduced by Szegedy, et al (2014):
  https://arxiv.org/pdf/1409.4842v1.pdf
  This function provides the pre-trained reimplementation from TF slim:
  https://github.com/tensorflow/models/tree/master/research/slim
  corresponding to the name "inception_v1".
  """

  model_path  = 'gs://modelzoo/InceptionV1_slim.pb'
  labels_path = 'gs://modelzoo/InceptionV1-labels.txt' #TODO
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

  model_path  = 'gs://modelzoo/InceptionV2_slim.pb'
  labels_path = 'gs://modelzoo/InceptionV1-labels.txt' #TODO
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

  model_path  = 'gs://modelzoo/InceptionV3_slim.pb'
  labels_path = 'gs://modelzoo/InceptionV1-labels.txt'
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

  model_path  = 'gs://modelzoo/InceptionV4_slim.pb'
  labels_path = 'gs://modelzoo/InceptionV1-labels.txt' #TODO
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

  model_path  = 'gs://modelzoo/InceptionResnetV2_slim.pb'
  labels_path = 'gs://modelzoo/InceptionV1-labels.txt' #TODO
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


class ResnetV2_50_slim(Model):
  """ResnetV2_50 as implemented by the TensorFlow slim framework.

  ResnetV2_50 was introduced by He, et al (2016):
  https://arxiv.org/pdf/1603.05027.pdf
  This function provides the pre-trained reimplementation from TF slim:
  https://github.com/tensorflow/models/tree/master/research/slim
  corresponding to the name "resnet_v2_50".
  """

  model_path  = 'gs://modelzoo/ResnetV2_50_slim.pb'
  labels_path = 'gs://modelzoo/InceptionV1-labels.txt' #TODO
  image_shape = [224, 224, 3]
  # inpute range taken from:
  # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py#L280
  image_value_range = (-1, 1)
  input_name = 'input'

  layers = [
     {'type': 'conv', 'name': 'resnet_v2_50/block1/unit_1/bottleneck_v2/preact/Relu', 'size': 64},
     {'type': 'conv', 'name': 'resnet_v2_50/block1/unit_1/bottleneck_v2/add', 'size': 256},
     {'type': 'conv', 'name': 'resnet_v2_50/block1/unit_2/bottleneck_v2/add', 'size': 256},
     {'type': 'conv', 'name': 'resnet_v2_50/block1/unit_3/bottleneck_v2/add', 'size': 256},
     {'type': 'conv', 'name': 'resnet_v2_50/block2/unit_1/bottleneck_v2/preact/Relu', 'size': 256},
     {'type': 'conv', 'name': 'resnet_v2_50/block2/unit_1/bottleneck_v2/add', 'size': 512},
     {'type': 'conv', 'name': 'resnet_v2_50/block2/unit_2/bottleneck_v2/add', 'size': 512},
     {'type': 'conv', 'name': 'resnet_v2_50/block2/unit_3/bottleneck_v2/add', 'size': 512},
     {'type': 'conv', 'name': 'resnet_v2_50/block2/unit_4/bottleneck_v2/add', 'size': 512},
     {'type': 'conv', 'name': 'resnet_v2_50/block3/unit_1/bottleneck_v2/preact/Relu', 'size': 512},
     {'type': 'conv', 'name': 'resnet_v2_50/block3/unit_1/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_50/block3/unit_2/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_50/block3/unit_3/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_50/block3/unit_4/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_50/block3/unit_5/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_50/block3/unit_6/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_50/block4/unit_1/bottleneck_v2/preact/Relu', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_50/block4/unit_1/bottleneck_v2/add', 'size': 2048},
     {'type': 'conv', 'name': 'resnet_v2_50/block4/unit_2/bottleneck_v2/add', 'size': 2048},
     {'type': 'conv', 'name': 'resnet_v2_50/block4/unit_3/bottleneck_v2/add', 'size': 2048},
     {'type': 'conv', 'name': 'resnet_v2_50/postnorm/Relu', 'size': 2048},
     {'type': 'dense', 'name': 'resnet_v2_50/predictions/Softmax', 'size': 1001},
   ]


class ResnetV2_101_slim(Model):
  """ResnetV2_101 as implemented by the TensorFlow slim framework.

  ResnetV2_101 was introduced by He, et al (2016):
  https://arxiv.org/pdf/1603.05027.pdf
  This function provides the pre-trained reimplementation from TF slim:
  https://github.com/tensorflow/models/tree/master/research/slim
  corresponding to the name "resnet_v2_101".
  """

  model_path  = 'gs://modelzoo/ResnetV2_101_slim.pb'
  labels_path = 'gs://modelzoo/InceptionV1-labels.txt' #TODO
  image_shape = [224, 224, 3]
  # inpute range taken from:
  # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py#L280
  image_value_range = (-1, 1)
  input_name = 'input'

  layers = [
     {'type': 'conv', 'name': 'resnet_v2_101/block1/unit_1/bottleneck_v2/preact/Relu', 'size': 64},
     {'type': 'conv', 'name': 'resnet_v2_101/block1/unit_1/bottleneck_v2/add', 'size': 256},
     {'type': 'conv', 'name': 'resnet_v2_101/block1/unit_2/bottleneck_v2/add', 'size': 256},
     {'type': 'conv', 'name': 'resnet_v2_101/block1/unit_3/bottleneck_v2/add', 'size': 256},
     {'type': 'conv', 'name': 'resnet_v2_101/block2/unit_1/bottleneck_v2/preact/Relu', 'size': 256},
     {'type': 'conv', 'name': 'resnet_v2_101/block2/unit_1/bottleneck_v2/add', 'size': 512},
     {'type': 'conv', 'name': 'resnet_v2_101/block2/unit_2/bottleneck_v2/add', 'size': 512},
     {'type': 'conv', 'name': 'resnet_v2_101/block2/unit_3/bottleneck_v2/add', 'size': 512},
     {'type': 'conv', 'name': 'resnet_v2_101/block2/unit_4/bottleneck_v2/add', 'size': 512},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_1/bottleneck_v2/preact/Relu', 'size': 512},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_1/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_2/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_3/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_4/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_5/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_6/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_7/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_8/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_9/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_10/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_11/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_12/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_13/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_14/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_15/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_16/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_17/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_18/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_19/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_20/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_21/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_22/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block3/unit_23/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block4/unit_1/bottleneck_v2/preact/Relu', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_101/block4/unit_1/bottleneck_v2/add', 'size': 2048},
     {'type': 'conv', 'name': 'resnet_v2_101/block4/unit_2/bottleneck_v2/add', 'size': 2048},
     {'type': 'conv', 'name': 'resnet_v2_101/block4/unit_3/bottleneck_v2/add', 'size': 2048},
     {'type': 'conv', 'name': 'resnet_v2_101/postnorm/Relu', 'size': 2048},
     {'type': 'dense', 'name': 'resnet_v2_101/predictions/Softmax', 'size': 1001},
   ]


class ResnetV2_152_slim(Model):
  """ResnetV2_152 as implemented by the TensorFlow slim framework.

  ResnetV2_152 was introduced by He, et al (2016):
  https://arxiv.org/pdf/1603.05027.pdf
  This function provides the pre-trained reimplementation from TF slim:
  https://github.com/tensorflow/models/tree/master/research/slim
  corresponding to the name "resnet_v2_152".
  """

  model_path  = 'gs://modelzoo/ResnetV2_152_slim.pb'
  labels_path = 'gs://modelzoo/InceptionV1-labels.txt' #TODO
  image_shape = [224, 224, 3]
  # inpute range taken from:
  # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py#L280
  image_value_range = (-1, 1)
  input_name = 'input'

  layers = [
     {'type': 'conv', 'name': 'resnet_v2_152/block1/unit_1/bottleneck_v2/preact/Relu', 'size': 64},
     {'type': 'conv', 'name': 'resnet_v2_152/block1/unit_1/bottleneck_v2/add', 'size': 256},
     {'type': 'conv', 'name': 'resnet_v2_152/block1/unit_2/bottleneck_v2/add', 'size': 256},
     {'type': 'conv', 'name': 'resnet_v2_152/block1/unit_3/bottleneck_v2/add', 'size': 256},
     {'type': 'conv', 'name': 'resnet_v2_152/block2/unit_1/bottleneck_v2/preact/Relu', 'size': 256},
     {'type': 'conv', 'name': 'resnet_v2_152/block2/unit_1/bottleneck_v2/add', 'size': 512},
     {'type': 'conv', 'name': 'resnet_v2_152/block2/unit_2/bottleneck_v2/add', 'size': 512},
     {'type': 'conv', 'name': 'resnet_v2_152/block2/unit_3/bottleneck_v2/add', 'size': 512},
     {'type': 'conv', 'name': 'resnet_v2_152/block2/unit_4/bottleneck_v2/add', 'size': 512},
     {'type': 'conv', 'name': 'resnet_v2_152/block2/unit_5/bottleneck_v2/add', 'size': 512},
     {'type': 'conv', 'name': 'resnet_v2_152/block2/unit_6/bottleneck_v2/add', 'size': 512},
     {'type': 'conv', 'name': 'resnet_v2_152/block2/unit_7/bottleneck_v2/add', 'size': 512},
     {'type': 'conv', 'name': 'resnet_v2_152/block2/unit_8/bottleneck_v2/add', 'size': 512},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_1/bottleneck_v2/preact/Relu', 'size': 512},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_1/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_2/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_3/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_4/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_5/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_6/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_7/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_8/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_9/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_10/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_11/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_12/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_13/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_14/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_15/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_16/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_17/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_18/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_19/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_20/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_21/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_22/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_23/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_24/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_25/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_26/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_27/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_28/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_29/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_30/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_31/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_32/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_33/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_34/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_35/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block3/unit_36/bottleneck_v2/add', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block4/unit_1/bottleneck_v2/preact/Relu', 'size': 1024},
     {'type': 'conv', 'name': 'resnet_v2_152/block4/unit_1/bottleneck_v2/add', 'size': 2048},
     {'type': 'conv', 'name': 'resnet_v2_152/block4/unit_2/bottleneck_v2/add', 'size': 2048},
     {'type': 'conv', 'name': 'resnet_v2_152/block4/unit_3/bottleneck_v2/add', 'size': 2048},
     {'type': 'conv', 'name': 'resnet_v2_152/postnorm/Relu', 'size': 2048},
     {'type': 'dense', 'name': 'resnet_v2_152/predictions/Softmax', 'size': 1001},
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
