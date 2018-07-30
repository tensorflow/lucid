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


IMAGENET_MEAN = np.asarray([123.68, 116.779, 103.939])
IMAGENET_MEAN_BGR = np.asarray([103.939, 116.779, 123.68])

def populate_inception_bottlenecks(scope):
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
    populate_inception_bottlenecks(scope)


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
  image_value_range = (-IMAGENET_MEAN, 255-IMAGENET_MEAN)
  input_name = 'data'


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
  # range based on emperical testing
  image_value_range = (-IMAGENET_MEAN, 255-IMAGENET_MEAN)
  input_name = 'data'


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
  image_value_range = (-IMAGENET_MEAN, 255-IMAGENET_MEAN)
  input_name = 'Placeholder'


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
  #TODO -- this model uses a BGR (!!!!!!) channel format!
  #        what do we do about that?
  model_path = 'gs://modelzoo/VGG16_caffe.pb'
  labels_path = 'gs://modelzoo/InceptionV1_caffe-labels.txt'
  image_shape = [224, 224, 3]
  image_value_range = (-IMAGENET_MEAN_BGR, 255-IMAGENET_MEAN_BGR)
  input_name = 'input'


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
  #TODO -- this model uses a BGR (!!!!!!) channel format!
  #        what do we do about that?
  model_path = 'gs://modelzoo/VGG19_caffe.pb'
  labels_path = 'gs://modelzoo/InceptionV1_caffe-labels.txt'
  image_shape = [224, 224, 3]
  image_value_range = (-IMAGENET_MEAN_BGR, 255-IMAGENET_MEAN_BGR)
  input_name = 'input'
