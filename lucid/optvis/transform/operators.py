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

"""Linear algebra helpers that create matrix operators/transformations."""

from __future__ import absolute_import, division, print_function

import numpy as np

def rotation(angleInRadians, axis):
    cos = np.cos(angleInRadians)
    sin = np.sin(angleInRadians)

    if axis == 0:
        raise NotImplementedError
    elif axis == 1:
        raise NotImplementedError
    elif axis == 2:
        return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    else:
        raise ValueError("Axis needs to be one of (0,1,2)!")


def deformation(x, y):
    matrix = np.identity(3, dtype=np.float32)
    matrix[0, 1] = x
    matrix[1, 0] = y
    return matrix


def shearing(angle, x, y):
    axis = 2 # z
    rotate = rotation(angle, axis)
    unrotate = rotation(-angle, axis)
    deform = deformation(x, y)
    return  np.matmul(unrotate, np.matmul(deform, rotate))


def translation(translation_x, translation_y):
    matrix = np.identity(3, dtype=np.float32)
    matrix[0, 2] = translation_x
    matrix[1, 2] = translation_y
    return matrix


def projection(vanishing_point_1, vanishing_point_2):
    matrix = np.identity(3, dtype=np.float32)
    matrix[2, 0] = vanishing_point_1
    matrix[2, 1] = vanishing_point_2
    return matrix


def homography(
    initial_translation, rotation, shear, projection, final_translation, shape
):
    center = translation(shape[0] // 2, shape[1] // 2)
    uncenter = translation(-shape[0] // 2, -shape[1] // 2)

    shear_centered = np.matmul(center, np.matmul(shear, uncenter))
    rotate_centered = np.matmul(center, np.matmul(rotation, uncenter))

    return (
        np.matmul(final_translation,
        np.matmul(projection,
        np.matmul(shear_centered,
        np.matmul(rotate_centered,
                  initial_translation))))
    )


def _parameterized_flattened_homography(
    translation1_x,
    translation1_y,
    rotationAngleInRadians,
    shearingAngleInRadians,
    shear_x,
    shear_y,
    vanishing_point_x,
    vanishing_point_y,
    translation2_x,
    translation2_y,
    shape_xy,
):
    initial_translate = translation(translation1_x, translation1_y)
    axis = 2  # z-axis: rotate a 2D image in its plane
    rotate = rotation(rotationAngleInRadians, axis)
    shear = shearing(shearingAngleInRadians, shear_x, shear_y)
    project = projection(vanishing_point_x, vanishing_point_y)
    final_translate = translation(translation2_x, translation2_y)
    matrix = homography(
        initial_translate, rotate, shear, project, final_translate, shape_xy
    )
    # return first 8 entries to conform to tf.contrib.image.transform intrerface
    return np.reshape(matrix, [-1])[:8].astype(np.float32)
