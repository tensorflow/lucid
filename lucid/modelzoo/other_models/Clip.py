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

from lucid.modelzoo.vision_base import Model

class Clip_ResNet50_4x(Model):
    image_value_range = (0, 255)
    input_name = 'input_image'
    model_name = "Clip_ResNet50_4x"
    image_shape = [288, 288, 3]
    model_path = "gs://modelzoo/vision/other_models/Clip_ResNet50_4x.pb"
