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

import tensorflow as tf
import numpy as np

from lucid.optvis.param.color import to_valid_rgb
from lucid.optvis.param.spatial import rfft2d_freqs


def image_sample(shape, decorrelate=True, sd=None, decay_power=1):
    raw_spatial = rand_fft_image(shape, sd=sd, decay_power=decay_power)
    return to_valid_rgb(raw_spatial, decorrelate=decorrelate)


# TODO: DRY with regard to fft_image from lucid.optvis.param.spatial
def rand_fft_image(shape, sd=None, decay_power=1):
    b, h, w, ch = shape
    sd = 0.01 if sd is None else sd

    imgs = []
    for _ in range(b):
        freqs = rfft2d_freqs(h, w)
        fh, fw = freqs.shape
        spectrum_var = sd * tf.random_normal([2, ch, fh, fw], dtype="float32")
        spectrum = tf.complex(spectrum_var[0], spectrum_var[1])
        spertum_scale = 1.0 / np.maximum(freqs, 1.0 / max(h, w)) ** decay_power
        # Scale the spectrum by the square-root of the number of pixels
        # to get a unitary transformation. This allows to use similar
        # learning rates to pixel-wise optimisation.
        spertum_scale *= np.sqrt(w * h)
        scaled_spectrum = spectrum * spertum_scale
        img = tf.spectral.irfft2d(scaled_spectrum)
        # in case of odd input dimension we cut off the additional pixel
        # we get from irfft2d length computation
        img = img[:ch, :h, :w]
        img = tf.transpose(img, [1, 2, 0])
        imgs.append(img)
    return tf.stack(imgs) / 4.0
