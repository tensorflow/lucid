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
from lucid.modelzoo.vision_base import Model


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

  This is a (re?)implementation of InceptionV1
  https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
  The weights were trained at Google and released in an early TensorFlow
  tutorial. It is possible the parameters are the original weights
  (trained in TensorFlow's predecessor), but we haven't been able to
  confirm this.

  As far as we can tell, it is exactly the same as the model described in
  the original paper, where as the slim and caffe implementations have
  minor implementation differences (such as eliding the heads).
  """
  model_path = 'gs://modelzoo/vision/other_models/InceptionV1.pb'
  labels_path = 'gs://modelzoo/labels/ImageNet_alternate.txt'
  dataset = 'ImageNet'
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
