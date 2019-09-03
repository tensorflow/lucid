from os.path import join
import json

import numpy as np
import tensorflow as tf
import scipy.ndimage as nd

import lucid.modelzoo.vision_models as models
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform
from lucid.misc.io import show, load, save
from lucid.misc.io.reading import read
from lucid.misc.ndimage_utils import resize
import lucid.misc.io.showing
from lucid.modelzoo.vision_base import SerializedModel


@objectives.wrap_objective()
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


def feature_inversion(*args, **kwargs):
  return caricature(*args, **kwargs)


def caricature(img, model, layer, n_steps=512, cossim_pow=0.0, verbose=True):
  if isinstance(layer, str):
    layers = [layer]
  elif isinstance(layer, (tuple, list)):
    layers = layer
  else:
    raise TypeError("layer must be str, tuple or list")

  with tf.Graph().as_default(), tf.Session() as sess:
    img = resize(img, model.image_shape[:2])

    objective = objectives.Objective.sum([
        1.0 * dot_compare(layer, cossim_pow=cossim_pow, batch=i+1)
        for i, layer in enumerate(layers)
    ])

    t_input = tf.placeholder(tf.float32, img.shape)
    param_f = param.image(img.shape[0], decorrelate=True, fft=True, alpha=False, batch=len(layers))
    param_f = tf.concat([t_input[None], param_f], 0)

    transforms = transform.standard_transforms + [transform.crop_or_pad_to(*model.image_shape[:2])]

    T = render.make_vis_T(model, objective, param_f, transforms=transforms)
    loss, vis_op, t_image = T("loss"), T("vis_op"), T("input")

    tf.global_variables_initializer().run()
    for i in range(n_steps): _ = sess.run([vis_op], {t_input: img})

    result = t_image.eval(feed_dict={t_input: img})

  if verbose:
    lucid.misc.io.showing.images(result[1:], layers)
  return result


def make_caricature(image_url, saved_model_folder_url, to, *args, **kwargs):
  image = load(image_url)
  model = SerializedModel.from_directory(saved_model_folder_url)
  layers = model.layer_names
  caricatures = caricature(image, model, layers, *args, verbose=False, **kwargs)

  results = {"type": "caricature"}

  save_input_url = join(to, "input.jpg")
  save(caricatures[0], save_input_url)
  results["input_image"] = save_input_url

  values_list = []
  for single_caricature, layer_name in zip(caricatures[1:], model.layer_names):
    save_caricature_url = join(to, layer_name + ".jpg")
    save(single_caricature, save_caricature_url)
    values_list.append({"type": "image", "url": save_caricature_url, "shape": single_caricature.shape})
  results["values"] = values_list

  save(results, join(to, "results.json"))

  return results
