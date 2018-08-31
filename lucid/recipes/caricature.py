import numpy as np
import tensorflow as tf
import scipy.ndimage as nd

import lucid.modelzoo.vision_models as models
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform
from lucid.misc.io import show, load
from lucid.misc.io.reading import read
import lucid.misc.io.showing

def imgToModelSize(arr, model):
  W = model.image_shape[0]
  w, h, _ = arr.shape
  s = float(W) / min(w,h)
  arr = nd.zoom(arr, [s, s, 1], mode="nearest")
  w, h, _ = arr.shape
  dw, dh = (w-W)//2, (h-W)//3
  return arr[dw:dw+W, dh:dh+W]
  
  
@objectives.wrap_objective
def dot_compare(layer, batch=1, cossim_pow=0):
  def inner(T):
    acts1 = T(layer)[0]
    acts2 = T(layer)[batch]
    dot = tf.reduce_sum(acts1 * acts2)
    mag = tf.sqrt(tf.reduce_sum(acts2**2))
    cossim = dot / (1e-6 + mag)
    cossim = tf.maximum(0.1, cossim)
    return dot * cossim ** cossim_pow
  return inner

def feature_inversion(img, model, layer, n_steps=512, cossim_pow=0.0, verbose=True):
  if isinstance(layer, str):
    layers = [layer]
  elif isinstance(layer, (tuple, list)):
    layers = layer
  else:
    raise TypeError("layer must be str, tuple or list")
  
  with tf.Graph().as_default(), tf.Session() as sess:
    img = imgToModelSize(img, model)
    
    objective = objectives.Objective.sum([
        1.0 * dot_compare(layer, cossim_pow=cossim_pow, batch=i+1)
        for i, layer in enumerate(layers)
    ])

    t_input = tf.placeholder(tf.float32, img.shape)
    param_f = param.image(img.shape[0], decorrelate=True, fft=True, alpha=False, batch=len(layers))
    param_f = tf.concat([t_input[None], param_f], 0)

    transforms = [
      transform.pad(8, mode='constant', constant_value=.5),
      transform.jitter(8),
      transform.random_scale([0.9, 0.95, 1.05, 1.1] + [1]*4),
      transform.random_rotate(list(range(-5, 5)) + [0]*5),
      transform.jitter(2),
    ]

    T = render.make_vis_T(model, objective, param_f, transforms=transforms)
    loss, vis_op, t_image = T("loss"), T("vis_op"), T("input")

    tf.global_variables_initializer().run()
    for i in range(n_steps): _ = sess.run([vis_op], {t_input: img})

    result = t_image.eval(feed_dict={t_input: img})
    if verbose:
      lucid.misc.io.showing.images(result[1:], layers)
    return result
