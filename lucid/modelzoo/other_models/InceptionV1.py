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
from lucid.modelzoo.vision_base import Model, _layers_from_list_of_dicts


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

InceptionV1.layers = _layers_from_list_of_dicts(InceptionV1, [
  {'tags': ['conv'], 'name': 'conv2d0', 'depth': 64},
  {'tags': ['conv'], 'name': 'conv2d1', 'depth': 64},
  {'tags': ['conv'], 'name': 'conv2d2', 'depth': 192},
  {'tags': ['conv'], 'name': 'mixed3a', 'depth': 256},
  {'tags': ['conv'], 'name': 'mixed3b', 'depth': 480},
  {'tags': ['conv'], 'name': 'mixed4a', 'depth': 508},
  {'tags': ['conv'], 'name': 'mixed4b', 'depth': 512},
  {'tags': ['conv'], 'name': 'mixed4c', 'depth': 512},
  {'tags': ['conv'], 'name': 'mixed4d', 'depth': 528},
  {'tags': ['conv'], 'name': 'mixed4e', 'depth': 832},
  {'tags': ['conv'], 'name': 'mixed5a', 'depth': 832},
  {'tags': ['conv'], 'name': 'mixed5b', 'depth': 1024},
  {'tags': ['conv'], 'name': 'head0_bottleneck', 'depth': 128},
  {'tags': ['dense'], 'name': 'nn0', 'depth': 1024},
  {'tags': ['dense'], 'name': 'softmax0', 'depth': 1008},
  {'tags': ['conv'], 'name': 'head1_bottleneck', 'depth': 128},
  {'tags': ['dense'], 'name': 'nn1', 'depth': 1024},
  {'tags': ['dense'], 'name': 'softmax1', 'depth': 1008},
  {'tags': ['dense'], 'name': 'softmax2', 'depth': 1008},
])
