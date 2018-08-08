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
from lucid.optvis.param.random import image_sample

def collapse_alpha(sd=0.5):
    def inner(t_image):
        assert t_image.get_shape().as_list()[-1] == 4
        rgb, a = t_image[..., :3], t_image[..., 3:4]
        rgb_shape = rgb.get_shape().as_list()
        rand_img = image_sample(rgb_shape, sd=sd)
        return a * rgb + (1 - a) * rand_img

    return inner
