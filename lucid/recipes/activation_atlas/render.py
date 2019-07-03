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
from itertools import chain


# TODO(schubert@): simplify, cleanup, dedupe objectives

import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform


@objectives.wrap_objective()
def direction_neuron_S(layer_name, vec, batch=None, x=None, y=None, S=None):

    def inner(T):
        layer = T(layer_name)
        shape = tf.shape(layer)
        x_ = shape[1] // 2 if x is None else x
        y_ = shape[2] // 2 if y is None else y
        if batch is None:
            raise RuntimeError("requires batch")

        acts = layer[batch, x_, y_]
        vec_ = vec
        if S is not None:
            vec_ = tf.matmul(vec_[None], S)[0]
        # mag = tf.sqrt(tf.reduce_sum(acts**2))
        dot = tf.reduce_mean(acts * vec_)
        # cossim = dot/(1e-4 + mag)
        return dot

    return inner


@objectives.wrap_objective()
def direction_neuron_cossim_S(
    layer_name, vec, batch=None, x=None, y=None, cossim_pow=2, S=None
):

    def inner(T):
        layer = T(layer_name)
        shape = tf.shape(layer)
        x_ = shape[1] // 2 if x is None else x
        y_ = shape[2] // 2 if y is None else y
        if batch is None:
            raise RuntimeError("requires batch")
            
        acts = layer[batch, x_, y_]
        vec_ = vec
        if S is not None:
            vec_ = tf.matmul(vec_[None], S)[0]
        mag = tf.sqrt(tf.reduce_sum(acts ** 2))
        dot = tf.reduce_mean(acts * vec_)
        cossim = dot / (1e-4 + mag)
        cossim = tf.maximum(0.1, cossim)
        return dot * cossim ** cossim_pow

    return inner


def render_icons(
    directions,
    model,
    layer,
    size=80,
    n_steps=128,
    verbose=False,
    S=None,
    num_attempts=3,
    cossim=True,
    alpha=False,
):

    model.load_graphdef()

    image_attempts = []
    loss_attempts = []

    depth = 4 if alpha else 3
    batch = len(directions)
    input_shape = (batch, size, size, depth)

    # Render two attempts, and pull the one with the lowest loss score.
    for attempt in range(num_attempts):

        # Render an image for each activation vector
        param_f = lambda: param.image(
            size, batch=len(directions), fft=True, decorrelate=True, alpha=alpha
        )

        if cossim is True:
            obj_list = [
                direction_neuron_cossim_S(layer, v, batch=n, S=S)
                for n, v in enumerate(directions)
            ]
        else:
            obj_list = [
                direction_neuron_S(layer, v, batch=n, S=S)
                for n, v in enumerate(directions)
            ]

        obj_list += [
          objectives.penalize_boundary_complexity(input_shape, w=5)
        ]

        obj = objectives.Objective.sum(obj_list)

        # holy mother of transforms
        transforms = [
           transform.pad(16, mode='constant'),
           transform.jitter(4),
           transform.jitter(4),
           transform.jitter(8),
           transform.jitter(8),
           transform.jitter(8),
           transform.random_scale(0.998**n for n in range(20,40)),
           transform.random_rotate(chain(range(-20,20), range(-10,10), range(-5,5), 5*[0])),
           transform.jitter(2),
           transform.crop_or_pad_to(size, size)
        ]
        if alpha:
            transforms.append(transform.collapse_alpha_random())

        # This is the tensorflow optimization process

        # print("attempt: ", attempt)
        with tf.Graph().as_default(), tf.Session() as sess:
            learning_rate = 0.05
            losses = []
            trainer = tf.train.AdamOptimizer(learning_rate)
            T = render.make_vis_T(model, obj, param_f, trainer, transforms)
            vis_op, t_image = T("vis_op"), T("input")
            losses_ = [obj_part(T) for obj_part in obj_list]
            tf.global_variables_initializer().run()
            for i in range(n_steps):
                loss, _ = sess.run([losses_, vis_op])
                losses.append(loss)
                # if i % 100 == 0:
                    # print(i)

            img = t_image.eval()
            img_rgb = img[:, :, :, :3]
            if alpha:
                # print("alpha true")
                k = 0.8
                bg_color = 0.0
                img_a = img[:, :, :, 3:]
                img_merged = img_rgb * ((1 - k) + k * img_a) + bg_color * k * (
                    1 - img_a
                )
                image_attempts.append(img_merged)
            else:
                # print("alpha false")
                image_attempts.append(img_rgb)

            loss_attempts.append(losses[-1])

    # Use only the icons with the lowest loss
    loss_attempts = np.asarray(loss_attempts)
    loss_final = []
    image_final = []
    # print("merging best scores from attempts...")
    for i, d in enumerate(directions):
        # note, this should be max, it is not a traditional loss
        mi = np.argmax(loss_attempts[:, i])
        image_final.append(image_attempts[mi][i])

    return (image_final, loss_final)
